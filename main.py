import numpy as np
import mediapipe as mp
import cv2
import os
import logging
import glob
import argparse
import math
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('face_morph')

# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def create_smooth_sequence(frames, interpolation_factor=3):
    """
    Use optical flow to create smoother transitions between frames
    
    Args:
        frames: Original sequence of morphed frames
        interpolation_factor: Number of frames to generate between each original frame
        
    Returns:
        List of frames with smooth transitions
    """
    if len(frames) < 2:
        return frames
        
    smooth_frames = [frames[0]]  # Start with the first frame
    
    # For each pair of consecutive frames
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        
        # Convert to grayscale for optical flow
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Generate intermediate frames
        for step in range(1, interpolation_factor + 1):
            # Calculate the fraction to move between frames
            t = step / (interpolation_factor + 1)
            
            # Create a grid of coordinates
            h, w = frame1.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            
            # Displacement field scaled by t
            x_displacement = x_coords + flow[..., 0] * t
            y_displacement = y_coords + flow[..., 1] * t
            
            # Ensure coordinates are within bounds
            x_displacement = np.clip(x_displacement, 0, w - 1)
            y_displacement = np.clip(y_displacement, 0, h - 1)
            
            # Remap pixels using the displacement field
            intermediate = cv2.remap(
                frame1, 
                x_displacement, 
                y_displacement, 
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            # Also remap from the other direction for better blending
            reverse_t = 1 - t
            
            # For reverse direction, the flow should be negated
            reverse_x = x_coords - flow[..., 0] * reverse_t
            reverse_y = y_coords - flow[..., 1] * reverse_t
            
            # Ensure coordinates are within bounds
            reverse_x = np.clip(reverse_x, 0, w - 1)
            reverse_y = np.clip(reverse_y, 0, h - 1)
            
            # Remap pixels from the second frame backward
            intermediate_reverse = cv2.remap(
                frame2, 
                reverse_x, 
                reverse_y, 
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            # Blend both directions for smoother transition
            intermediate_final = cv2.addWeighted(
                intermediate, 1 - t, 
                intermediate_reverse, t, 
                0
            )
            
            # Add to our sequence
            smooth_frames.append(intermediate_final)
        
        # Add the destination frame (except for the last iteration)
        if i < len(frames) - 2:
            smooth_frames.append(frame2)
    
    # Add the final frame
    smooth_frames.append(frames[-1])
    
    return smooth_frames

def apply_temporal_smoothing(frames, window_size=3):
    """
    Apply temporal smoothing to reduce flickering
    
    Args:
        frames: Sequence of video frames
        window_size: Size of the smoothing window (must be odd)
        
    Returns:
        Smoothed sequence of frames
    """
    if window_size < 3 or window_size % 2 == 0:
        window_size = 3  # Default to 3 if invalid
        
    radius = window_size // 2
    smoothed = frames.copy()
    
    # For each frame (except boundaries)
    for i in range(radius, len(frames) - radius):
        # Get window of frames
        window = frames[i - radius:i + radius + 1]
        
        # Calculate weighted average
        weights = np.array([1.0 / (abs(j - radius) + 1) for j in range(window_size)])
        weights = weights / weights.sum()
        
        # Apply weighted average to each channel
        result = np.zeros_like(frames[0], dtype=np.float32)
        for j, frame in enumerate(window):
            result += frame.astype(np.float32) * weights[j]
            
        smoothed[i] = result.astype(np.uint8)
        
    return smoothed

def _detect_raw(image):
    """Run MediaPipe on the given image at downscaled resolution. Returns landmarks scaled to original size, or None."""
    h, w = image.shape[:2]
    max_detect_w = 1024
    if w > max_detect_w:
        scale = max_detect_w / w
        detect_img = cv2.resize(image, (max_detect_w, int(h * scale)))
    else:
        detect_img = image

    image_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        return [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]


def _rotate_bound(image, angle):
    """Rotate image by `angle` degrees (positive = counter-clockwise) and expand
    the canvas so nothing is cropped. Returns (rotated_image, transform_matrix)."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
    return rotated, M


def _detect_at_angle(image, angle):
    """Pre-rotate by `angle` degrees, run MediaPipe, return landmarks mapped back
    to the original image's coordinate system, or None on detection failure."""
    if angle == 0:
        return _detect_raw(image)

    rotated, M = _rotate_bound(image, angle)
    pts = _detect_raw(rotated)
    if pts is None:
        return None

    M_inv = cv2.invertAffineTransform(M)
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    pts_back = cv2.transform(pts_arr, M_inv).reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in pts_back]


def _face_likeness_score(pts):
    """Heuristic: how 'face-shaped' are these detected landmarks?
    Lower = better. Uses anatomy-aware checks that fail when MediaPipe returned
    a rotated/skewed mesh. We assume index 1 = nose tip, 10 = forehead, 152 = chin,
    33 = right eye outer, 263 = left eye outer.
    """
    if pts is None or len(pts) < 455:
        return float("inf")
    pts = np.asarray(pts, dtype=np.float32)
    nose = pts[1]
    forehead = pts[10]
    chin = pts[152]
    r_eye = pts[33]
    l_eye = pts[263]

    eye_axis = l_eye - r_eye
    eye_len = np.linalg.norm(eye_axis)
    face_axis = chin - forehead
    face_len = np.linalg.norm(face_axis)
    if eye_len < 1 or face_len < 1:
        return float("inf")

    # On an upright face, eye axis is roughly perpendicular to forehead→chin axis.
    cos_angle = abs(np.dot(eye_axis, face_axis)) / (eye_len * face_len)
    perpendicularity_penalty = cos_angle  # 0 when perfectly perpendicular

    # Nose tip should sit between eye-line and mouth-line — i.e., roughly halfway
    # along forehead→chin axis (project nose onto that axis, expect ~0.4–0.7).
    t = float(np.dot(nose - forehead, face_axis) / (face_len ** 2))
    nose_position_penalty = abs(t - 0.55)

    return perpendicularity_penalty + nose_position_penalty


def _find_face_bbox(image):
    """Use MediaPipe FaceDetection (full-range model) to locate a face. Returns
    (x, y, w, h) in original-image pixel coords or None. Tries multiple scales —
    the full-range model has resolution sweet spots and may miss tiny faces in
    very large images."""
    h, w = image.shape[:2]
    target_widths = sorted({w, 1024, 640, 480}, reverse=True)
    for tw in target_widths:
        if tw > w:
            continue
        if tw < w:
            test_img = cv2.resize(image, (tw, int(h * tw / w)))
        else:
            test_img = image
        rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as fd:
            results = fd.process(rgb)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                logger.info(f"FaceDetection found bbox at scale w={tw}")
                return (
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int(bbox.width * w),
                    int(bbox.height * h),
                )
    return None


def _load_manual_bbox(image_path, w, h):
    """Read <image_path>.bbox.json (normalized {x, y, w, h} in [0..1]) if present."""
    if image_path is None:
        return None
    import json
    side = f"{image_path}.bbox.json"
    if not os.path.exists(side):
        return None
    with open(side) as f:
        b = json.load(f)
    return (int(b["x"] * w), int(b["y"] * h), int(b["w"] * w), int(b["h"] * h))


def _detect_via_crop(image, sweep_angles, image_path=None):
    """Locate a small/distant face via FaceDetection or manual bbox sidecar,
    crop with margin, run FaceMesh + angle sweep on the crop, map landmarks
    back to full-image coords."""
    h, w = image.shape[:2]
    bbox = _load_manual_bbox(image_path, w, h)
    if bbox is not None:
        logger.info(f"Using manual bbox from {image_path}.bbox.json: {bbox}")
    else:
        bbox = _find_face_bbox(image)
    if bbox is None:
        return None
    fx, fy, fw, fh = bbox

    # Expand bbox by 60% to give FaceMesh room around the face
    margin = 0.6
    cx = fx + fw / 2
    cy = fy + fh / 2
    side = max(fw, fh) * (1 + margin)
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(w, int(cx + side / 2))
    y1 = min(h, int(cy + side / 2))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # Try native + each angle on the crop, keep best by face-likeness
    candidates = []
    for angle in [0] + list(sweep_angles):
        pts = _detect_at_angle(crop, angle) if angle != 0 else _detect_raw(crop)
        if pts is None:
            continue
        score = _face_likeness_score(pts)
        candidates.append((score, angle, pts))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    score, angle, pts = candidates[0]
    logger.info(f"Detected via face-detection crop ({angle:+}°, score={score:.3f}, "
                f"crop=({x0},{y0})-({x1},{y1}))")

    # Map crop-local coords back to original image coords
    return [(x + x0, y + y0) for (x, y) in pts]


def _detect_with_rotation(image, image_path=None):
    """Try MediaPipe on the original. If that fails, sweep rotation angles and
    pick the one whose landmarks look most face-shaped. A sidecar `<path>.angle.txt`
    overrides the sweep with a fixed angle."""
    # 1. Manual angle override
    if image_path:
        side = f"{image_path}.angle.txt"
        if os.path.exists(side):
            with open(side) as f:
                forced = float(f.read().strip())
            logger.info(f"Forcing rotation angle {forced}° from {side}")
            return _detect_at_angle(image, forced)

    # 2. Native first — most reliable when it works
    pts = _detect_raw(image)
    if pts is not None:
        return pts

    # 3. Sweep candidate angles, pick best by face-likeness score
    sweep_angles = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
    candidates = []
    for angle in sweep_angles:
        pts = _detect_at_angle(image, angle)
        if pts is None:
            continue
        score = _face_likeness_score(pts)
        candidates.append((score, angle, pts))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        score, angle, pts = candidates[0]
        logger.info(f"Detected via {angle:+}° rotation (score={score:.3f}; tried {len(candidates)} angles)")
        return pts

    # 4. Last resort: use FaceDetection (better at distant faces) to crop, then mesh
    return _detect_via_crop(image, sweep_angles, image_path=image_path)


def _load_manual_landmarks(image_path, w, h):
    """Load manual landmark override from <image_path>.landmarks.json.

    JSON format (all coords are NORMALIZED [0..1]):
      {"landmarks": [[x, y], ...]}            # full 478-point list
      OR
      {"sparse": {"<idx>": [x, y], ...}}      # partial — only the indices used by SELECTED_INDICES
                                              # missing entries default to image center
    """
    import json
    side = f"{image_path}.landmarks.json"
    if not os.path.exists(side):
        return None
    with open(side) as f:
        data = json.load(f)

    def to_pixel(p):
        return [int(p[0] * w), int(p[1] * h)]

    if "landmarks" in data:
        return np.array([to_pixel(p) for p in data["landmarks"]], dtype=np.int32)
    if "sparse" in data:
        n_pts = max(SELECTED_INDICES) + 1
        pts = np.tile(np.array([w // 2, h // 2]), (n_pts, 1)).astype(np.int32)
        for k, v in data["sparse"].items():
            pts[int(k)] = to_pixel(v)
        return pts
    return None


# Function to get facial landmarks from an image
def get_landmarks(image, image_path=None):
    h, w = image.shape[:2]

    if image_path:
        manual = _load_manual_landmarks(image_path, w, h)
        if manual is not None:
            logger.info(f"Using manual landmarks from {image_path}.landmarks.json")
            return manual

    pts_list = _detect_with_rotation(image, image_path=image_path)
    if pts_list is None:
        return np.array([])
    return np.array(pts_list)

# Function to create triangles using Delaunay triangulation
def get_delaunay_triangles(rect, points):
    # Create subdivision for Delaunay triangulation
    subdiv = cv2.Subdiv2D(rect)
    
    # Add points to subdivision
    for point in points:
        subdiv.insert((point[0], point[1]))
    
    # Get list of triangles
    triangles = subdiv.getTriangleList()
    
    # Convert triangles to point indices
    delaunay_triangles = []
    for triangle in triangles:
        pt1 = (int(triangle[0]), int(triangle[1]))
        pt2 = (int(triangle[2]), int(triangle[3]))
        pt3 = (int(triangle[4]), int(triangle[5]))
        
        # Find indices of points in the original array
        # Added error checking to handle exact matching issues
        index1 = find_point_index(points, pt1)
        index2 = find_point_index(points, pt2)
        index3 = find_point_index(points, pt3)
        
        if index1 is not None and index2 is not None and index3 is not None:
            delaunay_triangles.append([index1, index2, index3])
    
    return delaunay_triangles


# Helper function to find point index with tolerance
def find_point_index(points, point, tolerance=1):
    for i, pt in enumerate(points):
        if abs(pt[0] - point[0]) <= tolerance and abs(pt[1] - point[1]) <= tolerance:
            return i
    return None

# Function to morph one triangle
def morph_triangle(img1, img2, img_morph, t1, t2, t_morph, alpha):
    # Find bounding rectangles of triangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r_morph = cv2.boundingRect(np.float32([t_morph]))
    
    # Offset triangle coordinates for correct ROI handling
    t1_rect = []
    t2_rect = []
    t_morph_rect = []
    
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t_morph_rect.append(((t_morph[i][0] - r_morph[0]), (t_morph[i][1] - r_morph[1])))
    
    # Create mask for triangle
    mask = np.zeros((r_morph[3], r_morph[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_morph_rect), (1.0, 1.0, 1.0), 16, 0)
    
    # Apply affine transformations
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    
    # Check if the rectangles are valid
    if img1_rect.size == 0 or img2_rect.size == 0:
        return
    
    size_morph = (r_morph[2], r_morph[3])
    
    # Make sure we have at least 3 points for affine transform
    if len(t1_rect) < 3 or len(t2_rect) < 3 or len(t_morph_rect) < 3:
        return
    
    # Ensure all points are unique for affine transform
    if len(set(tuple(map(tuple, t1_rect)))) < 3 or len(set(tuple(map(tuple, t2_rect)))) < 3 or len(set(tuple(map(tuple, t_morph_rect)))) < 3:
        return
    
    # Affine transformations for morphing
    try:
        warp_mat1 = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t_morph_rect))
        warp_mat2 = cv2.getAffineTransform(np.float32(t2_rect), np.float32(t_morph_rect))
        
        # Apply affine transformations
        img1_warped = cv2.warpAffine(img1_rect, warp_mat1, size_morph, None, 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img2_warped = cv2.warpAffine(img2_rect, warp_mat2, size_morph, None, 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Blend images
        img_rect = (1.0 - alpha) * img1_warped + alpha * img2_warped
        
        # Copy blended triangle to result
        # Ensure the ROI is within bounds
        y_start = r_morph[1]
        y_end = r_morph[1] + r_morph[3]
        x_start = r_morph[0]
        x_end = r_morph[0] + r_morph[2]
        
        h, w = img_morph.shape[:2]
        
        # Clip to image boundaries
        y_start = max(0, min(y_start, h-1))
        y_end = max(0, min(y_end, h))
        x_start = max(0, min(x_start, w-1))
        x_end = max(0, min(x_end, w))
        
        # Only apply if we have a valid ROI
        if y_end > y_start and x_end > x_start:
            roi = img_morph[y_start:y_end, x_start:x_end]
            mask_roi = mask[:y_end-y_start, :x_end-x_start]
            img_rect_roi = img_rect[:y_end-y_start, :x_end-x_start]
            
            # Apply the blending only if shapes match
            if roi.shape == mask_roi.shape == img_rect_roi.shape and roi.size > 0:
                img_morph[y_start:y_end, x_start:x_end] = roi * (1 - mask_roi) + img_rect_roi * mask_roi
    except Exception as e:
        logger.error(f"Error in morph_triangle: {e}")
        # Skip this triangle if there's an error

FACE_OUTLINE_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
NOSE_INDICES = [1, 2, 3, 4, 5, 6, 19, 129, 49, 131, 134, 51, 195, 197, 237]
MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
SELECTED_INDICES = sorted(set(FACE_OUTLINE_INDICES + LEFT_EYE_INDICES + RIGHT_EYE_INDICES + NOSE_INDICES + MOUTH_INDICES))


def prepare_morph(img1, img2, path1=None, path2=None):
    """Detect landmarks and compute triangulation ONCE on mean shape."""
    landmarks1 = get_landmarks(img1, path1)
    landmarks2 = get_landmarks(img2, path2)

    min_needed = max(SELECTED_INDICES) + 1
    if landmarks1 is None or landmarks2 is None or len(landmarks1) < min_needed or len(landmarks2) < min_needed:
        return None

    h = int((img1.shape[0] + img2.shape[0]) / 2)
    w = int((img1.shape[1] + img2.shape[1]) / 2)

    boundary = np.array([
        [0, 0], [0, h-1], [w-1, 0], [w-1, h-1],
        [w//2, 0], [w//2, h-1], [0, h//2], [w-1, h//2]
    ])

    pts1 = np.vstack([np.array([landmarks1[i] for i in SELECTED_INDICES]), boundary])
    pts2 = np.vstack([np.array([landmarks2[i] for i in SELECTED_INDICES]), boundary])

    # Triangulate once on the mean shape — same topology for all frames
    pts_mean = ((pts1.astype(np.float32) + pts2.astype(np.float32)) / 2).astype(np.float32)
    triangles = get_delaunay_triangles((0, 0, w, h), pts_mean)

    return pts1, pts2, triangles, h, w


def apply_morph_frame(img1, img2, pts1, pts2, triangles, h, w, alpha):
    """Render one morphed frame using pre-computed triangulation."""
    pts_morph = ((1 - alpha) * pts1 + alpha * pts2).astype(np.float32)
    img_morph = np.zeros((h, w, 3), dtype=np.float32)

    for tri in triangles:
        t1 = [pts1[tri[i]] for i in range(3)]
        t2 = [pts2[tri[i]] for i in range(3)]
        t_m = [pts_morph[tri[i]] for i in range(3)]
        morph_triangle(img1, img2, img_morph, t1, t2, t_m, alpha)

    if np.mean(img_morph) < 5.0:
        return cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    return np.uint8(img_morph)


def generate_morph_sequence(img1, img2, steps=10, path1=None, path2=None):
    """Creates a sequence of morphed frames with fixed triangulation and cosine easing."""
    morph_data = prepare_morph(img1, img2, path1, path2)

    if morph_data is None:
        logger.warning("Could not detect landmarks, using fallback blend")
        results = []
        for i in range(steps + 1):
            t = i / steps
            alpha = (1 - math.cos(math.pi * t)) / 2
            results.append(cv2.addWeighted(img1, 1-alpha, img2, alpha, 0))
        return results

    pts1, pts2, triangles, h, w = morph_data
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    results = []

    for i in range(steps + 1):
        t = i / steps
        alpha = (1 - math.cos(math.pi * t)) / 2
        logger.info(f"Processing morph frame {i}/{steps} (alpha={alpha:.2f})")
        results.append(apply_morph_frame(img1, img2, pts1, pts2, triangles, h, w, alpha))

    return results

def create_morph_video(frames, output_dir, fps=24, pause_frames=15, loops=1, 
                     interpolation_factor=3, apply_smoothing=True, temporal_window=3):
    """
    Create a video from the morphing sequence with improved codec handling
    """
    if not frames:
        logger.error("No frames provided for video creation")
        return
    
    logger.info(f"Starting video creation with {len(frames)} original frames")
    
    # Apply frame interpolation for smoother transitions
    if interpolation_factor > 0:
        logger.info(f"Applying optical flow interpolation (factor: {interpolation_factor})...")
        frames = create_smooth_sequence(frames, interpolation_factor)
        logger.info(f"Generated {len(frames)} frames after interpolation")
    
    # Apply temporal smoothing if enabled
    if apply_smoothing and len(frames) > temporal_window:
        logger.info(f"Applying temporal smoothing (window size: {temporal_window})...")
        frames = apply_temporal_smoothing(frames, temporal_window)
    
    # Get dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # List of codecs to try, in order of preference
    codecs = [
        ('mp4v', 'morph_video.mp4'),         # Default MP4 codec
        ('avc1', 'morph_video_h264.mp4'),    # H.264 codec (alternative name)
        ('H264', 'morph_video_h264.mp4'),    # H.264 codec
        ('XVID', 'morph_video_xvid.avi'),    # XVID codec (widely supported)
        ('MJPG', 'morph_video_mjpg.avi'),    # Motion JPEG
    ]
    
    # Try each codec until one works
    success = False
    for codec, filename in codecs:
        try:
            video_path = os.path.join(output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Check if codec is supported by creating a small test video
            test_path = os.path.join(output_dir, f"test_{filename}")
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            
            # If the writer isn't initialized properly, OpenCV might not raise an error but will return False
            if not test_writer.isOpened():
                logger.warning(f"Codec '{codec}' not properly initialized, trying next codec...")
                test_writer.release()
                continue
                
            test_writer.write(frames[0])  # Write a test frame
            test_writer.release()
            
            # If we got here, codec is supported
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Create smooth forward and backward transition with pauses
            for _ in range(loops):
                # Forward transition (with pause at beginning)
                for _ in range(pause_frames):
                    video_writer.write(frames[0])
                    
                # Actual morph sequence
                for frame in frames:
                    video_writer.write(frame)
                                
            # # Final pause at first frame
            # for _ in range(pause_frames):
            #     video_writer.write(frames[1])
                
            # Release the video writer
            video_writer.release()
            logger.info(f"Successfully created morphing video with '{codec}' codec: {video_path}")
            success = True
            break  # Stop trying codecs if this one worked
            
        except Exception as e:
            logger.warning(f"Failed to create video with codec '{codec}': {e}")
            continue
    
    if not success:
        logger.error("Failed to create video with any codec. Try installing additional codec support.")
        
        # Fallback: save as individual frames in an 'animation' subfolder
        animation_dir = os.path.join(output_dir, 'animation')
        if not os.path.exists(animation_dir):
            os.makedirs(animation_dir)
            
        logger.info(f"Saving individual frames to {animation_dir}")
        for i, frame in enumerate(frames):
            frame_path = os.path.join(animation_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(frame_path, frame)
        
        logger.info(f"Saved {len(frames)} frames as individual images")


def _build_effect_lookup(effects, total_frames):
    """Expand each {frame, type, magnitude} effect into a per-frame lookup
    {video_frame: (zoom_factor, color_intensity)}. Effects span ~7 frames
    centered on the beat frame, with a smooth bell envelope so the punch
    eases in and out without a jump cut."""
    import math as _m
    lookup = {}
    n_frames = 7
    half = n_frames // 2
    for eff in effects:
        center = int(eff["frame"])
        magnitude = float(eff.get("magnitude", 0.10))
        is_color = eff.get("type") == "zoom_punch_color"
        for offset in range(-half, half + 1):
            f = center + offset
            if f < 0 or f >= total_frames:
                continue
            # cos² bell: 1 at center, 0 at edges
            bell = _m.cos(offset / (half + 1) * _m.pi / 2) ** 2
            zoom = 1.0 + magnitude * bell
            color = (magnitude * bell) if is_color else 0.0
            prev_zoom, prev_color = lookup.get(f, (1.0, 0.0))
            lookup[f] = (max(prev_zoom, zoom), max(prev_color, color))
    return lookup


def _apply_zoom(frame, zoom):
    if zoom <= 1.001:
        return frame
    h, w = frame.shape[:2]
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    crop = frame[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_color_pop(frame, intensity):
    if intensity <= 0.001:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + intensity * 1.5), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + intensity * 1.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def create_morph_video_from_paths(frame_paths, output_dir, fps=24, pause_frames=15, loops=1,
                               interpolation_factor=3, apply_smoothing=True, temporal_window=3,
                               tail_pause=0, effects=None):
    """
    Create a video from the morphing sequence using file paths instead of keeping frames in memory
    Supports optical flow interpolation and temporal smoothing with a batched approach
    """
    if not frame_paths:
        logger.error("No frame paths provided for video creation")
        return
    
    logger.info(f"Starting video creation with {len(frame_paths)} frames")
    
    # Load the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        logger.error(f"Failed to load frame: {frame_paths[0]}")
        return
        
    height, width = first_frame.shape[:2]
    
    # Create processed frames directory if using effects
    processed_dir = None
    if interpolation_factor > 0 or (apply_smoothing and len(frame_paths) > temporal_window):
        processed_dir = os.path.join(output_dir, 'processed_frames')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
    
    # Process frames with effects if needed
    final_frame_paths = frame_paths
    
    # Apply frame interpolation for smoother transitions
    if interpolation_factor > 0:
        logger.info(f"Applying optical flow interpolation (factor: {interpolation_factor})...")
        
        # Process frames in batches to manage memory
        batch_size = 5  # Small batch size for optical flow processing
        new_frame_paths = []
        
        # For each small batch of frames
        for i in range(0, len(frame_paths) - 1, batch_size - 1):
            batch_paths = frame_paths[i:i + batch_size]
            
            # Load batch into memory
            batch_frames = []
            for path in batch_paths:
                frame = cv2.imread(path)
                if frame is not None:
                    batch_frames.append(frame)
                else:
                    logger.warning(f"Failed to load frame: {path}")
            
            if len(batch_frames) < 2:
                continue
                
            # Apply optical flow interpolation to this batch
            interpolated_frames = create_smooth_sequence(batch_frames, interpolation_factor)
            
            # Save interpolated frames to disk
            for j, frame in enumerate(interpolated_frames):
                # Determine if this is the first frame of a batch (except first batch)
                is_duplicate = j == 0 and i > 0
                
                # Skip duplicates at batch boundaries
                if not is_duplicate:
                    out_path = os.path.join(processed_dir, f'interp_{len(new_frame_paths):04d}.jpg')
                    cv2.imwrite(out_path, frame)
                    new_frame_paths.append(out_path)
        
        logger.info(f"Generated {len(new_frame_paths)} frames after interpolation")
        final_frame_paths = new_frame_paths
    
    # Apply temporal smoothing if enabled
    if apply_smoothing and len(final_frame_paths) > temporal_window:
        logger.info(f"Applying temporal smoothing (window size: {temporal_window})...")
        
        # Process in larger batches for temporal smoothing
        batch_size = temporal_window * 4  # Larger batch size as each window needs multiple frames
        smoothed_frame_paths = []
        
        # For each batch with overlap
        for i in range(0, len(final_frame_paths), batch_size - temporal_window * 2):
            # Get batch with padding for window operations
            start_idx = max(0, i - temporal_window)
            end_idx = min(len(final_frame_paths), i + batch_size + temporal_window)
            batch_paths = final_frame_paths[start_idx:end_idx]
            
            # Load batch into memory
            batch_frames = []
            for path in batch_paths:
                frame = cv2.imread(path)
                if frame is not None:
                    batch_frames.append(frame)
                else:
                    logger.warning(f"Failed to load frame: {path}")
                    # Use a placeholder frame if loading fails
                    if batch_frames:
                        batch_frames.append(batch_frames[-1])
                    else:
                        batch_frames.append(first_frame)
            
            # Apply temporal smoothing
            smoothed_frames = apply_temporal_smoothing(batch_frames, temporal_window)
            
            # Determine which frames to save (skip padding frames)
            save_start = 0 if i == 0 else temporal_window
            save_end = len(smoothed_frames) if i + batch_size >= len(final_frame_paths) else -temporal_window
            
            # Save smoothed frames
            for j, frame in enumerate(smoothed_frames[save_start:save_end]):
                out_path = os.path.join(processed_dir, f'smooth_{len(smoothed_frame_paths):04d}.jpg')
                cv2.imwrite(out_path, frame)
                smoothed_frame_paths.append(out_path)
        
        logger.info(f"Generated {len(smoothed_frame_paths)} frames after smoothing")
        final_frame_paths = smoothed_frame_paths
    
    # List of codecs to try, in order of preference
    codecs = [
        ('mp4v', 'morph_video.mp4'),         # Default MP4 codec
        ('avc1', 'morph_video_h264.mp4'),    # H.264 codec (alternative name)
        ('H264', 'morph_video_h264.mp4'),    # H.264 codec
        ('XVID', 'morph_video_xvid.avi'),    # XVID codec (widely supported)
        ('MJPG', 'morph_video_mjpg.avi'),    # Motion JPEG
    ]
    
    # Try each codec until one works
    success = False
    for codec, filename in codecs:
        try:
            video_path = os.path.join(output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Check if codec is supported
            test_path = os.path.join(output_dir, f"test_{filename}")
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            
            if not test_writer.isOpened():
                logger.warning(f"Codec '{codec}' not properly initialized, trying next codec...")
                test_writer.release()
                continue
                
            test_writer.write(first_frame)  # Write a test frame
            test_writer.release()
            
            # If we got here, codec is supported
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            total_frames = pause_frames + len(final_frame_paths) + tail_pause
            effect_lookup = _build_effect_lookup(effects or [], total_frames)

            def write_with_effects(frame, video_frame_idx):
                if video_frame_idx in effect_lookup:
                    zoom, color = effect_lookup[video_frame_idx]
                    frame = _apply_zoom(frame, zoom)
                    frame = _apply_color_pop(frame, color)
                video_writer.write(frame)

            # Create smooth forward and backward transition with pauses
            for _ in range(loops):
                video_frame_idx = 0
                # Forward transition (with pause at beginning)
                first_frame = cv2.imread(final_frame_paths[0])
                for _ in range(pause_frames):
                    write_with_effects(first_frame, video_frame_idx)
                    video_frame_idx += 1

                # Process frames in batches to save memory
                batch_size = 10
                for i in range(0, len(final_frame_paths), batch_size):
                    batch_paths = final_frame_paths[i:i+batch_size]
                    for path in batch_paths:
                        frame = cv2.imread(path)
                        if frame is None:
                            logger.warning(f"Failed to load frame: {path}")
                            frame = first_frame
                        write_with_effects(frame, video_frame_idx)
                        video_frame_idx += 1

                # Tail hold on the last frame so the final image lingers for `tail_pause` frames
                if tail_pause > 0:
                    last_frame = cv2.imread(final_frame_paths[-1])
                    if last_frame is not None:
                        for _ in range(tail_pause):
                            write_with_effects(last_frame, video_frame_idx)
                            video_frame_idx += 1

            # Release the video writer
            video_writer.release()
            logger.info(f"Successfully created morphing video with '{codec}' codec: {video_path}")
            success = True
            break  # Stop trying codecs if this one worked
            
        except Exception as e:
            logger.warning(f"Failed to create video with codec '{codec}': {e}")
            continue
    
    if not success:
        logger.error("Failed to create video with any codec. Try installing additional codec support.")
        
    # Clean up temporary directory if it was created
    if processed_dir and os.path.exists(processed_dir):
        logger.info(f"Cleaning up temporary processed frames directory: {processed_dir}")
        try:
            import shutil
            shutil.rmtree(processed_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")

def process_multiple_images(images, output_dir, steps_between=10, start_index=0, image_paths=None):
    """
    Process multiple images to create a morphing sequence between all of them
    Supports resuming from a specific pair index
    
    Args:
        images: List of loaded images
        output_dir: Directory to save results
        steps_between: Number of morphing steps between each pair of images
        start_index: Index of the pair to start processing from (for resuming)
    
    Returns:
        List of paths to all frames in the complete morphing sequence
    """
    if len(images) < 2:
        logger.error("Need at least 2 images to create a morphing sequence")
        return []
    
    # Ensure start_index is valid
    if start_index < 0 or start_index >= len(images) - 1:
        logger.warning(f"Invalid start_index: {start_index}. Setting to 0.")
        start_index = 0
    
    # Calculate average dimensions
    avg_height = int(sum(img.shape[0] for img in images) / len(images))
    avg_width = int(sum(img.shape[1] for img in images) / len(images))
    
    # Resize all images to the same dimensions
    resized_images = [cv2.resize(img, (avg_width, avg_height)) for img in images]
    
    # Create pairs of consecutive resized images, with optional source paths
    if image_paths is None:
        image_paths = [None] * len(images)
    all_pairs = [
        (resized_images[i], resized_images[i+1], image_paths[i], image_paths[i+1])
        for i in range(len(resized_images)-1)
    ]
    
    # Directory to save intermediate results
    pairs_dir = os.path.join(output_dir, 'pairs')
    if not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir)
    
    # Directory to save the complete sequence
    sequence_dir = os.path.join(output_dir, 'sequence')
    if not os.path.exists(sequence_dir):
        os.makedirs(sequence_dir)
    
    # Instead of storing all frames in memory, just keep track of their paths
    all_frame_paths = []
    
    # Count how many frames we should have in total up to the start_index
    expected_frames_before_start = start_index * steps_between + 1 if start_index > 0 else 0
    
    # If resuming, check if the frame files exist and collect their paths
    if start_index > 0:
        logger.info(f"Resuming from pair {start_index+1}/{len(all_pairs)}")
        
        # Find existing frames in the sequence directory
        existing_frames = sorted(glob.glob(os.path.join(sequence_dir, 'frame_*.jpg')))
        
        if len(existing_frames) == expected_frames_before_start:
            logger.info(f"Found {len(existing_frames)} existing frames")
            all_frame_paths = existing_frames
        else:
            logger.warning(f"Existing frame count ({len(existing_frames)}) doesn't match expected count ({expected_frames_before_start}). Will regenerate from the beginning.")
            all_frame_paths = []
            start_index = 0
    
    logger.info(f"Processing {len(all_pairs) - start_index} remaining image pairs...")
    
    # Keep track of the next frame index
    next_frame_index = len(all_frame_paths)
    
    # Process only the pairs we need to
    for i, (img1, img2, path1, path2) in enumerate(all_pairs[start_index:], start=start_index):
        logger.info(f"Morphing pair {i+1}/{len(all_pairs)}")

        # Generate morph sequence for this pair
        morph_frames = generate_morph_sequence(img1, img2, steps=steps_between, path1=path1, path2=path2)
        
        # Save the morphed frames
        for j, frame in enumerate(morph_frames):
            frame_path = os.path.join(pairs_dir, f'pair_{i+1}_frame_{j:02d}.jpg')
            cv2.imwrite(frame_path, frame)
            
            # Save to the sequence directory if needed
            if i == 0 or j > 0:  # For the first pair, add all frames; for others, skip first frame
                sequence_path = os.path.join(sequence_dir, f'frame_{next_frame_index:04d}.jpg')
                cv2.imwrite(sequence_path, frame)
                all_frame_paths.append(sequence_path)
                next_frame_index += 1
                
        # Clear the morph_frames to free memory
        morph_frames = None
        
    logger.info(f"Generated a total of {len(all_frame_paths)} frames")
    
    return all_frame_paths


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Morphing for Multiple Images')
    parser.add_argument('--input', '-i', type=str, default='images/*.jpg', help='Input images glob pattern')
    parser.add_argument('--output', '-o', type=str, default='morph_results', help='Output directory')
    parser.add_argument('--steps', '-s', type=int, default=60, help='Number of steps between each pair of images')
    parser.add_argument('--fps', '-f', type=int, default=50, help='FPS for output video')
    parser.add_argument('--interpolation', type=int, default=0, help='Optical flow interpolation factor')
    parser.add_argument('--smoothing', action='store_true', help='Apply temporal smoothing')
    parser.add_argument('--loops', type=int, default=1, help='Number of video loops')
    parser.add_argument('--pause', type=int, default=15, help='Number of pause frames at endpoints')
    parser.add_argument('--tail', type=int, default=0, help='Frames to hold the last image at the end')
    parser.add_argument('--effects', type=str, default=None, help='Path to effects.json with [{frame, type, magnitude}, ...]')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--resume', type=int, default=0, help='Resume from this pair index (0-based)')
    args = parser.parse_args()
    
    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Get list of input image paths
    image_paths = sorted(glob.glob(args.input))
    
    if not image_paths:
        logger.error(f"No images found matching pattern: {args.input}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load images, keeping paths in lock-step for manual-landmark lookup
    images = []
    loaded_paths = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            loaded_paths.append(path)
            logger.info(f"Loaded: {path}, shape: {img.shape}")
        else:
            logger.error(f"Failed to load image: {path}")
    
    if len(images) < 2:
        logger.error("Need at least 2 valid images to create a morphing sequence")
        return
    
    # Check resume index validity
    if args.resume >= len(images) - 1:
        logger.warning(f"Resume index {args.resume} is out of range. Setting to 0.")
        args.resume = 0
    
    # Process multiple images with resume capability
    all_frame_paths = process_multiple_images(
        images,
        args.output,
        steps_between=args.steps,
        start_index=args.resume,
        image_paths=loaded_paths,
    )
    
    # Load effects spec if provided
    effects = None
    if args.effects:
        import json
        with open(args.effects) as f:
            effects = json.load(f).get("effects", [])
        logger.info(f"Loaded {len(effects)} effects from {args.effects}")

    # Create video from frames on disk
    if all_frame_paths:
        # Modified create_morph_video function call
        create_morph_video_from_paths(
            all_frame_paths,
            args.output,
            fps=args.fps,
            pause_frames=args.pause,
            loops=args.loops,
            interpolation_factor=args.interpolation,
            apply_smoothing=args.smoothing,
            temporal_window=3,
            tail_pause=args.tail,
            effects=effects,
        )
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()

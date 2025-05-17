import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image


def generate_morph_sequence_with_beats(beat_file, img1, img2, points1, points2, tri_list, size, output, beat_start=None, beat_end=None):
    """
    Generate a morphing sequence synchronized with beats.
    
    Args:
        beat_file (str): Path to the CSV file with beat information
        img1, img2, points1, points2, tri_list, size: Same as in original function
        output (str): Output video path
        beat_start: Starting beat information for this segment
        beat_end: Ending beat information for this segment
    """
    import csv
    import numpy as np
    from PIL import Image
    from subprocess import Popen, PIPE

    # Read beat information from CSV
    beats = []
    with open(beat_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            beats.append({
                'timestamp': float(row['timestamp']),
                'strength': float(row['strength'])
            })
    
    if not beats:
        print("No beats found in the file. Exiting.")
        return
    
    # Calculate video parameters
    if beat_start and beat_end:
        print(f"Using specific beat segment: {beat_start['beat_number']} to {beat_end['beat_number']}")
        # Calculate duration from the specific beats
        segment_duration = beat_end['timestamp'] - beat_start['timestamp']
        # Add a small buffer to ensure we reach the end beat
        total_duration = segment_duration * 1.05
    else:
        # Calculate video parameters from the whole beat file
        total_duration = beats[-1]['timestamp']
        # Add a bit extra to the end
        total_duration += min(1.0, beats[-1]['timestamp'] - beats[-2]['timestamp'] if len(beats) > 1 else 1.0)
    
    print(f"total duration: {total_duration}")
    # Frame rate should be high enough for smooth transitions
    frame_rate = 30
    num_frames = int(total_duration * frame_rate)    
    # Start ffmpeg process
    p = Popen([
        'ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),
        '-s', f"{size[1]}x{size[0]}", '-i', '-', 
        '-c:v', 'libx264', '-crf', '25',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-pix_fmt', 'yuv420p', output
    ], stdin=PIPE)
    
    # Convert images to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Function to find alpha value at a given time
    def get_alpha_at_time(time):
        # Handle specific beat segment if provided
        if beat_start and beat_end:
            # Calculate adjusted time relative to the start beat
            adjusted_time = beat_start['timestamp'] + time
            
            # Check if we're outside the range
            if adjusted_time <= beat_start['timestamp']:
                return 0.0
            if adjusted_time >= beat_end['timestamp']:
                return 1.0
                
            # Linear interpolation between start and end beat
            progress = (adjusted_time - beat_start['timestamp']) / (beat_end['timestamp'] - beat_start['timestamp'])
            return progress
        
        # Standard beat processing for the whole file
        # Find the nearest beats before and after the current time
        prev_beat = next((b for b in reversed(beats) if b['timestamp'] <= time), beats[0])
        next_beat = next((b for b in beats if b['timestamp'] > time), beats[-1])
        
        # If exactly on a beat, use its strength as a factor
        if abs(time - prev_beat['timestamp']) < 0.001:
            # Exactly on a beat, use a more pronounced effect
            return prev_beat['strength']
        
        # Calculate linear progression between beats
        if prev_beat['timestamp'] == next_beat['timestamp']:
            # Handle edge case
            return 0.5
            
        progress = (time - prev_beat['timestamp']) / (next_beat['timestamp'] - prev_beat['timestamp'])
        
        # Linear interpolation between beats
        return progress


    # Generate frames
    for frame_num in range(num_frames):
        # Calculate current time
        current_time = frame_num / frame_rate
        
        # Calculate alpha for this frame
        alpha = get_alpha_at_time(current_time)
        
        # Ensure alpha stays within bounds
        alpha = max(0.0, min(1.0, alpha))
        
        # Compute weighted average point coordinates
        points = []
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))
        
        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)
        
        # Morph triangles
        for i in range(len(tri_list)):
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]
            
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
        
        # Save the frame to the video
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(p.stdin, 'JPEG')
        
        # Print progress occasionally
        if frame_num % 30 == 0:
            print(f"Processed frame {frame_num}/{num_frames} at time {current_time:.2f}s (alpha: {alpha:.2f})")
    
    # Close the ffmpeg process
    p.stdin.close()
    p.wait()
    print(f"Morphing complete. Output saved to: {output}")


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def generate_morph_sequence(duration,frame_rate,img1,img2,points1,points2,tri_list,size,output):

    num_images = int(duration*frame_rate)
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', output], stdin=PIPE)
    
    for j in range(0, num_images):

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points = []
        alpha = j/(num_images-1)

        # Compute weighted average point coordinates
        #print(len(points1))
        #print(len(points2))
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        
        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype = img1.dtype)

        for i in range(len(tri_list)):    
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time.
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
            
            pt1 = (int(t[0][0]), int(t[0][1]))
            pt2 = (int(t[1][0]), int(t[1][1]))
            pt3 = (int(t[2][0]), int(t[2][1]))

            #cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            #cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            #cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)
            
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save(p.stdin,'JPEG')

    p.stdin.close()
    p.wait()
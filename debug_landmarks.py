"""Visualize detected landmarks on a single image to debug detection issues.

Usage:
    python debug_landmarks.py path/to/image.jpg [output.jpg]

Saves an image with all detected landmarks drawn as small dots, plus larger
labelled markers on the SELECTED_INDICES that the morph pipeline actually uses.
"""
import sys
import cv2
import numpy as np

# Reuse the detection pipeline from main.py
import main as m


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{img_path}.debug.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        sys.exit(1)

    landmarks = m.get_landmarks(img, image_path=img_path)
    if landmarks is None or len(landmarks) == 0:
        print("No landmarks detected (even with rotation fallback)")
        sys.exit(1)

    print(f"Got {len(landmarks)} landmarks")

    canvas = img.copy()
    for x, y in landmarks:
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 200, 0), -1)

    # Larger markers on points actually used downstream
    selected = m.SELECTED_INDICES
    for i in selected:
        if i >= len(landmarks):
            continue
        x, y = landmarks[i]
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), 2)

    # Highlight key anatomical points
    key_points = {
        1: "nose_tip",
        10: "forehead",
        152: "chin",
        33: "R_eye_outer",
        263: "L_eye_outer",
        61: "mouth_R",
        291: "mouth_L",
    }
    for idx, name in key_points.items():
        if idx >= len(landmarks):
            continue
        x, y = landmarks[idx]
        cv2.circle(canvas, (int(x), int(y)), 8, (255, 0, 255), 2)
        cv2.putText(canvas, name, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)

    cv2.imwrite(out_path, canvas)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

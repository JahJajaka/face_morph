"""Draw a bounding box around a face on an image; saves <image>.bbox.json.

Usage:
    python bbox_tool.py path/to/image.jpg

Drag a rectangle around the face. Press ENTER or SPACE to confirm; c/ESC to cancel.
The pipeline reads this sidecar and runs FaceMesh on the cropped region, which
helps when faces are too small/distant for MediaPipe's automatic detector.
"""
import sys
import os
import cv2
import json


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    max_w, max_h = 1400, 900
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        display = img

    print("Drag a rectangle around the face, then press ENTER (or SPACE).")
    roi = cv2.selectROI("Select face bbox", display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        print("No selection. Aborting.")
        sys.exit(1)

    dh, dw = display.shape[:2]
    out = {
        "x": rx / dw,
        "y": ry / dh,
        "w": rw / dw,
        "h": rh / dh,
    }
    out_path = f"{img_path}.bbox.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}: {out}")


if __name__ == "__main__":
    main()

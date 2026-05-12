"""Manual landmark annotation tool for images that fail MediaPipe detection.

Usage:
    python annotate.py path/to/image.jpg

Click each requested point. Keys:
    space / right-arrow : skip current point (defaults to image center)
    b / left-arrow      : go back one point
    s                   : save and continue
    q / ESC             : save and quit

Saves coordinates to <image>.landmarks.json with normalized [0..1] coords
in 'sparse' format keyed by MediaPipe landmark index. main.py picks this up
automatically via _load_manual_landmarks().
"""
import cv2
import json
import os
import sys

FACE_OUTLINE_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
NOSE_INDICES = [1, 2, 3, 4, 5, 6, 19, 129, 49, 131, 134, 51, 195, 197, 237]
MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

LABELS = {
    10:  "Top of forehead (centerline)",
    152: "Chin tip (bottom)",
    234: "Right cheek (face's rightmost point near ear)",
    454: "Left cheek (face's leftmost point near ear)",
    33:  "Right eye outer corner",
    133: "Right eye inner corner",
    362: "Left eye inner corner",
    263: "Left eye outer corner",
    1:   "Nose tip",
    61:  "Right corner of mouth",
    291: "Left corner of mouth",
    13:  "Top of upper lip (centerline)",
    17:  "Bottom of lower lip (centerline)",
    0:   "Upper lip cupid's bow (centerline)",
    54:  "Right brow outer end",
    284: "Left brow outer end",
}


def build_click_list():
    """Walk through SELECTED_INDICES grouped by region for a logical click order."""
    seen = set()
    click_list = []

    def add(idx, region, label=None):
        if idx in seen:
            return
        seen.add(idx)
        if label is None:
            label = LABELS.get(idx, f"{region} point #{idx}")
        click_list.append((idx, region, label))

    # Anchor points first — distinctive easy-to-find features
    for i in [10, 152, 234, 454, 33, 263, 1, 61, 291, 13, 17, 0, 133, 362, 54, 284]:
        if i in LABELS:
            add(i, "anchor", LABELS[i])

    # Then the rest, grouped
    for i in FACE_OUTLINE_INDICES:
        add(i, "face outline")
    for i in RIGHT_EYE_INDICES:
        add(i, "right eye")
    for i in LEFT_EYE_INDICES:
        add(i, "left eye")
    for i in NOSE_INDICES:
        add(i, "nose")
    for i in MOUTH_INDICES:
        add(i, "mouth")

    return click_list


def fit_to_screen(image, max_w=1400, max_h=900):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(image, (int(w * scale), int(h * scale))), scale
    return image, 1.0


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        sys.exit(1)

    display, _ = fit_to_screen(image)
    disp_h, disp_w = display.shape[:2]

    out_path = f"{image_path}.landmarks.json"
    sparse = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        sparse = data.get("sparse", {})
        print(f"Loaded {len(sparse)} existing points from {out_path}")

    click_list = build_click_list()

    cur = [0]
    last_click = [None]

    def redraw():
        canvas = display.copy()
        for k, (nx, ny) in sparse.items():
            x = int(float(nx) * disp_w)
            y = int(float(ny) * disp_h)
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

        if cur[0] < len(click_list):
            idx, region, label = click_list[cur[0]]
            existing = sparse.get(str(idx))
            if existing is not None:
                ex = int(existing[0] * disp_w)
                ey = int(existing[1] * disp_h)
                cv2.circle(canvas, (ex, ey), 7, (0, 255, 255), 2)
            header = f"[{cur[0]+1}/{len(click_list)}] {region}: {label}  (idx {idx})"
        else:
            header = f"All done! Press 's' to save, 'b' to revisit, 'q' to save+quit"

        cv2.rectangle(canvas, (0, 0), (disp_w, 30), (0, 0, 0), -1)
        cv2.putText(canvas, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("annotate", canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and cur[0] < len(click_list):
            idx, _, _ = click_list[cur[0]]
            sparse[str(idx)] = [x / disp_w, y / disp_h]
            last_click[0] = cur[0]
            cur[0] += 1
            redraw()

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", on_mouse)
    redraw()

    def save():
        with open(out_path, "w") as f:
            json.dump({"sparse": sparse}, f, indent=2)
        print(f"Saved {len(sparse)} points to {out_path}")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (ord('q'), 27):  # q or ESC
            save()
            break
        elif key == ord('s'):
            save()
        elif key in (ord('b'), 81):  # b or left arrow
            if cur[0] > 0:
                cur[0] -= 1
                redraw()
        elif key in (ord(' '), 83):  # space or right arrow
            if cur[0] < len(click_list):
                cur[0] += 1
                redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

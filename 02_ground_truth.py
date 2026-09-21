"""
Step 2: Create Ground Truth masks (since none were provided)
Method: GrabCut (semi-automatic) seeded with a rectangle that assumes
the fruit (object of interest) is roughly centered, with a margin from
the border. This gives a reasonable approximate ground-truth mask for
"object (positive=1) vs background (negative=0)" without manual pixel
labeling of 50 images by hand.

Output: gt/<name>_gt.png  (0/255 binary mask, 255 = fruit/foreground)
"""
import os
import cv2
import numpy as np

IMG_DIR = "images"
GT_DIR = "gt"
os.makedirs(GT_DIR, exist_ok=True)

MARGIN_RATIO = 0.06  # rectangle margin from border as fraction of image size

files = sorted(os.listdir(IMG_DIR))
for f in files:
    img = cv2.imread(os.path.join(IMG_DIR, f))
    h, w = img.shape[:2]

    mx = int(w * MARGIN_RATIO)
    my = int(h * MARGIN_RATIO)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    fallback = np.zeros((h, w), np.uint8)
    fallback[my:h - my, mx:w - mx] = 255

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
        gt = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_ratio = gt.mean() / 255.0
        # GrabCut sometimes degenerates on images where the fruit fills almost
        # the whole frame (no real background to learn from). In that case its
        # foreground estimate becomes too small/too large to be usable, so we
        # fall back to "everything inside the rectangle is foreground".
        if fg_ratio < 0.15 or fg_ratio > 0.97:
            print(f"{f}: GrabCut degenerate (fg={fg_ratio:.2f}) -> fallback rect")
            gt = fallback
    except cv2.error as e:
        print("GrabCut failed on", f, "-> fallback to full rect", e)
        gt = fallback

    # light cleanup: fill small holes, keep largest component
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(gt)
        cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
        gt = clean

    out_name = f.replace(".jpg", "_gt.png")
    cv2.imwrite(os.path.join(GT_DIR, out_name), gt)

print(f"Ground truth masks created for {len(files)} images -> {GT_DIR}/")
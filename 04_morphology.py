"""
Step 4: Improve segmentation using Morphological operations
(directly applying Lecture 10 concepts)

Pipeline applied to the RAW mask from Step 3:
  1. Opening   (Erosion -> Dilation)  : removes small noisy speckles /
                                         thin protrusions scattered over
                                         the background (salt noise).
  2. Closing   (Dilation -> Erosion)  : fills small holes inside the
                                         fruit region and joins narrow
                                         gaps/breaks in the object.
  3. Keep the LARGEST connected component (the fruit is one dominant
     blob; leftover small blobs elsewhere are false positives).
  4. Fill remaining internal holes (flood-fill from the border).

Structuring element: elliptical SE, size scaled relative to image size
(so the amount of "shrink/grow" is consistent across different image
resolutions), matching the "Structuring Element" idea in the lecture.
"""
import os
import cv2
import numpy as np

IMG_DIR = "images"
RAW_DIR = "pred_raw"
OUT_DIR = "pred_morph"
os.makedirs(OUT_DIR, exist_ok=True)


def remove_small_components(mask, min_area_ratio=0.004):
    """Remove connected components smaller than `min_area_ratio` of the
    total image area. Unlike keeping only the single largest blob, this
    correctly handles images with several separate fruit instances
    (e.g. scattered cherries / strawberries) while still discarding tiny
    speckle noise left over after Opening/Closing."""
    h, w = mask.shape
    min_area = min_area_ratio * h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros_like(mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == lbl] = 255
    return keep


def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return mask | flood_inv


files = sorted(os.listdir(IMG_DIR))
for f in files:
    base = f.replace(".jpg", "")
    img = cv2.imread(os.path.join(IMG_DIR, f))
    h, w = img.shape[:2]
    raw = cv2.imread(os.path.join(RAW_DIR, base + "_raw.png"), cv2.IMREAD_GRAYSCALE)

    # structuring element size scales with image size (~1.5% of the
    # smaller dimension), minimum 3x3
    k = max(3, int(min(h, w) * 0.015))
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, se)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se)

    cleaned = remove_small_components(closed)
    cleaned = fill_holes(cleaned)

    cv2.imwrite(os.path.join(OUT_DIR, base + "_morph.png"), cleaned)

print(f"Morphological clean-up (Opening -> Closing -> largest CC -> fill holes) done for {len(files)} images.")

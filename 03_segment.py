"""
Step 3: Segmentation algorithm (before morphology)

Idea:
  1. Estimate the background color by sampling a thin frame around the
     border of the image (assumption: the border mostly shows background).
  2. Convert image to CIE Lab color space (perceptually closer to human
     vision than RGB) and compute, for every pixel, its color distance
     to the estimated background color -> "score map".
     High score = very different from background = likely FOREGROUND (fruit).
  3. Threshold the score map automatically with Otsu's method to obtain
     a binary mask: 1 = predicted fruit (Positive), 0 = predicted background.

This binary mask is the RAW segmentation result, i.e. what a "median
filter"-style / simple thresholding approach gives us -- BEFORE we apply
any morphological clean-up (that is Step 4).
"""
import os
import cv2
import numpy as np

IMG_DIR = "images"
SCORE_DIR = "scores"
RAW_DIR = "pred_raw"
os.makedirs(SCORE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

BORDER_FRAC = 0.03  # thickness of border ring used to estimate background color


def estimate_background_lab(lab_img):
    h, w = lab_img.shape[:2]
    b = max(2, int(min(h, w) * BORDER_FRAC))
    ring_pixels = np.concatenate([
        lab_img[:b, :, :].reshape(-1, 3),
        lab_img[-b:, :, :].reshape(-1, 3),
        lab_img[:, :b, :].reshape(-1, 3),
        lab_img[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(ring_pixels, axis=0)


files = sorted(os.listdir(IMG_DIR))
for f in files:
    img = cv2.imread(os.path.join(IMG_DIR, f))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_color = estimate_background_lab(lab)
    diff = lab - bg_color.reshape(1, 1, 3)
    score = np.sqrt((diff ** 2).sum(axis=2))  # Euclidean distance in Lab space

    # normalize score to 0-255 for Otsu thresholding / saving
    score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_thr, raw_mask = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    base = f.replace(".jpg", "")
    np.save(os.path.join(SCORE_DIR, base + "_score.npy"), score_norm)
    cv2.imwrite(os.path.join(RAW_DIR, base + "_raw.png"), raw_mask)

print(f"Raw segmentation (score maps + Otsu masks) done for {len(files)} images.")

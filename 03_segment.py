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

# --- ตั้งค่าพื้นฐาน ---
IMG_DIR = "images"     # ดึงภาพ input จากโฟลเดอร์นี้ (output ของ 01_prepare.py)
SCORE_DIR = "scores"   # เก็บ score map แบบต่อเนื่อง (.npy) ไว้ที่นี่ -> ใช้ทำ ROC curve ใน 05_evaluate.py
RAW_DIR = "pred_raw"   # เก็บ mask ขาว-ดำหลัง threshold ไว้ที่นี่ -> ใช้เป็น input ให้ 04_morphology.py
os.makedirs(SCORE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

BORDER_FRAC = 0.03  # thickness of border ring used to estimate background color
                     # -> ความหนาของแถบขอบภาพที่ใช้ "เดาสีพื้นหลัง" (3% ของด้านสั้น)


def estimate_background_lab(lab_img):
    # รับภาพ (Lab space) มา 1 ภาพ แล้วคืนค่า "สีพื้นหลังโดยประมาณ" (ค่า L,a,b 1 ค่า)
    # วิธีคิด: ดึงเฉพาะพิกเซลที่อยู่ในแถบขอบภาพ (บน+ล่าง+ซ้าย+ขวา) มารวมกัน
    # แล้วหาค่ามัธยฐาน (median) ของแถบขอบทั้งหมด -> ใช้เป็นตัวแทนสีพื้นหลัง
    # ที่ใช้ "ขอบภาพ" เพราะ dataset นี้ผลไม้มักอยู่กลางภาพ ขอบภาพจึงน่าจะเป็นพื้นหลังจริง
    h, w = lab_img.shape[:2]
    b = max(2, int(min(h, w) * BORDER_FRAC))   # ความหนาแถบขอบ (pixel)
    ring_pixels = np.concatenate([
        lab_img[:b, :, :].reshape(-1, 3),      # แถบขอบบน
        lab_img[-b:, :, :].reshape(-1, 3),     # แถบขอบล่าง
        lab_img[:, :b, :].reshape(-1, 3),      # แถบขอบซ้าย
        lab_img[:, -b:, :].reshape(-1, 3),     # แถบขอบขวา
    ], axis=0)
    return np.median(ring_pixels, axis=0)      # ค่ามัธยฐานของสีในแถบขอบทั้ง 4 ด้าน


# ดึงรายชื่อไฟล์ทั้งหมดจาก images/ (มาจาก output ของ 01_prepare.py)
files = sorted(os.listdir(IMG_DIR))
for f in files:
    img = cv2.imread(os.path.join(IMG_DIR, f))          # อ่านภาพสี (BGR) จาก images/<f>
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)  # แปลงเป็นปริภูมิสี Lab

    bg_color = estimate_background_lab(lab)              # เรียกฟังก์ชันด้านบน -> ได้สีพื้นหลัง 1 ค่า
    diff = lab - bg_color.reshape(1, 1, 3)                # หาผลต่างสีของทุกพิกเซล เทียบกับสีพื้นหลัง
    score = np.sqrt((diff ** 2).sum(axis=2))  # Euclidean distance in Lab space
    # -> score คือ "ระยะห่างสี" ต่อพิกเซล ยิ่งมากยิ่งต่างจากพื้นหลัง = น่าจะเป็นผลไม้

    # normalize score to 0-255 for Otsu thresholding / saving
    score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # -> ปรับสเกลค่า score ให้อยู่ในช่วง 0-255 (เพื่อเซฟเป็นภาพ/ใช้ Otsu ได้)

    # หา threshold อัตโนมัติด้วย Otsu's method บน score_norm แล้วตัดเป็นภาพขาวดำ (raw_mask)
    otsu_thr, raw_mask = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    base = f.replace(".jpg", "")
    # เซฟ score map แบบละเอียด (.npy) -> ใช้ตอนทำ ROC curve ใน 05_evaluate.py
    np.save(os.path.join(SCORE_DIR, base + "_score.npy"), score_norm)
    # เซฟ mask ขาว-ดำ (คำตอบสุดท้ายหลัง threshold) -> ใช้เป็น input ให้ 04_morphology.py
    cv2.imwrite(os.path.join(RAW_DIR, base + "_raw.png"), raw_mask)

print(f"Raw segmentation (score maps + Otsu masks) done for {len(files)} images.")

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
# NOTE (คอมเมนต์เสริม ไม่ใช่ของเดิม): ข้อ 3 ใน docstring ด้านบนเขียนว่า
# "Keep the LARGEST connected component" แต่โค้ดจริงด้านล่างเรียก
# remove_small_components() ซึ่งเก็บ "ทุกก้อนที่ใหญ่พอ" ไว้ทั้งหมด
# ไม่ได้เก็บแค่ก้อนที่ใหญ่ที่สุดก้อนเดียว (ดู docstring ของฟังก์ชันนั้นด้านล่าง
# อธิบายเหตุผลไว้แล้วว่าทำไมถึงเปลี่ยนวิธีคิด) — docstring บนสุดนี้เป็นของเดิม
# ที่ยังไม่ได้อัปเดตให้ตรงกับพฤติกรรมปัจจุบันของโค้ด
import os
import cv2
import numpy as np

# --- ตั้งค่าพื้นฐาน ---
IMG_DIR = "images"      # ใช้แค่ดึงขนาดภาพ (h, w) ไม่ได้ใช้สีของภาพเลย
RAW_DIR = "pred_raw"    # ดึง mask ขาว-ดำจริงที่ต้องประมวลผล (output จาก 03_segment.py)
OUT_DIR = "pred_morph"  # ผลลัพธ์หลัง morphology จะเซฟไว้ที่นี่ -> ใช้ต่อใน 05_evaluate.py
os.makedirs(OUT_DIR, exist_ok=True)


def remove_small_components(mask, min_area_ratio=0.004):
    """Remove connected components smaller than `min_area_ratio` of the
    total image area. Unlike keeping only the single largest blob, this
    correctly handles images with several separate fruit instances
    (e.g. scattered cherries / strawberries) while still discarding tiny
    speckle noise left over after Opening/Closing."""
    h, w = mask.shape
    min_area = min_area_ratio * h * w   # พื้นที่ขั้นต่ำที่ถือว่า "ไม่ใช่ noise" (คำนวณจากสัดส่วนพื้นที่ภาพ)
    # หาก้อนที่เชื่อมต่อกัน (connected components) ทั้งหมดใน mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask   # ไม่มีก้อนเลย (ทั้งภาพเป็นพื้นหลัง) -> คืนค่าเดิมไปเลย
    keep = np.zeros_like(mask)
    for lbl in range(1, n):   # lbl=0 คือพื้นหลัง ข้ามไป เริ่มที่ label 1
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:   # เก็บเฉพาะก้อนที่พื้นที่ >= เกณฑ์
            keep[labels == lbl] = 255
    return keep


def fill_holes(mask):
    # เติมรูดำที่อยู่ "ข้างใน" ก้อนขาว โดยใช้เทคนิค flood-fill จากมุม (0,0) ของภาพ
    # แนวคิด: flood-fill จากขอบภาพจะไหลเข้าไปทุกที่ที่เป็นพื้นหลังที่ต่อถึงขอบได้
    # ส่วนที่ flood-fill "ไปไม่ถึง" (รูที่ถูกล้อมรอบด้วยก้อนขาวสนิท) คือรูที่ต้องเติม
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)   # mask เสริมที่ cv2.floodFill ต้องการ (ใหญ่กว่าเดิม +2 ทุกด้าน)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)      # ไล่เติมสีขาวจากมุมภาพ (0,0) ไปตามพื้นที่ดำที่ต่อกัน
    flood_inv = cv2.bitwise_not(flood)              # กลับสี -> ตอนนี้เหลือแต่ "รูที่ถูกล้อมรอบ" เป็นสีขาว
    return mask | flood_inv                          # รวม mask เดิม + รูที่เติมแล้ว = ผลลัพธ์สุดท้าย


# ดึงรายชื่อไฟล์จาก images/ (ใช้แค่กำหนดว่าจะประมวลผลภาพไหนบ้าง วนตามชื่อเดียวกับ pred_raw/)
files = sorted(os.listdir(IMG_DIR))
for f in files:
    base = f.replace(".jpg", "")
    img = cv2.imread(os.path.join(IMG_DIR, f))   # อ่านภาพสีจาก images/<f> (เอาแค่ขนาดภาพไปใช้ ด้านล่าง)
    h, w = img.shape[:2]
    # อ่าน mask ขาว-ดำ (ผลจาก Otsu ใน 03_segment.py) จาก pred_raw/<base>_raw.png
    raw = cv2.imread(os.path.join(RAW_DIR, base + "_raw.png"), cv2.IMREAD_GRAYSCALE)

    # structuring element size scales with image size (~1.5% of the
    # smaller dimension), minimum 3x3
    k = max(3, int(min(h, w) * 0.015))   # ขนาด SE ผูกกับขนาดภาพ (h, w ที่ดึงมาจาก images/ ด้านบน)
    if k % 2 == 0:
        k += 1   # บังคับให้เป็นเลขคี่เสมอ (OpenCV structuring element ต้องมีจุดศูนย์กลางชัดเจน)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))   # สร้าง SE รูปวงรีขนาด k x k

    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, se)    # Opening: ลบจุด noise เล็กๆ บนพื้นหลัง
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se)  # Closing: เติมรูเล็กๆ/เชื่อมรอยขาดในตัววัตถุ

    cleaned = remove_small_components(closed)   # เรียกฟังก์ชันด้านบน: ลบก้อนที่เล็กเกินไป (noise ที่เหลือ)
    cleaned = fill_holes(cleaned)                # เรียกฟังก์ชันด้านบน: เติมรูที่เหลือให้เต็ม

    # เซฟผลลัพธ์สุดท้ายหลัง morphology -> pred_morph/<base>_morph.png
    cv2.imwrite(os.path.join(OUT_DIR, base + "_morph.png"), cleaned)

print(f"Morphological clean-up (Opening -> Closing -> largest CC -> fill holes) done for {len(files)} images.")

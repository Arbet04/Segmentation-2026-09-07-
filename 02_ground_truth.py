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

# --- ตั้งค่าพื้นฐาน ---
IMG_DIR = "images"   # ดึงภาพ input จากโฟลเดอร์นี้ (ต้องมาจาก 01_prepare.py มาก่อนแล้ว)
GT_DIR = "gt"         # ผลลัพธ์ ground-truth mask จะเซฟไว้ที่นี่
os.makedirs(GT_DIR, exist_ok=True)   # สร้างโฟลเดอร์ gt/ ถ้ายังไม่มี

MARGIN_RATIO = 0.06  # rectangle margin from border as fraction of image size
                      # -> ตัวเลขนี้กำหนดว่า "กรอบเริ่มต้น" ที่บอก GrabCut ว่าวัตถุน่าจะอยู่ตรงไหน
                      #    เว้นขอบจากทุกด้าน 6% ของขนาดภาพ (สมมติว่าวัตถุอยู่กลางภาพ)

# ดึงรายชื่อไฟล์ทั้งหมดจาก images/ (มาจาก output ของ 01_prepare.py)
files = sorted(os.listdir(IMG_DIR))
for f in files:
    img = cv2.imread(os.path.join(IMG_DIR, f))  # อ่านภาพสีจริงจาก images/<f>
    h, w = img.shape[:2]                          # ขนาดภาพ (สูง, กว้าง)

    # คำนวณกรอบสี่เหลี่ยม (rect) ที่ใช้เป็น "จุดเริ่มต้น" ให้ GrabCut เดาว่าข้างในกรอบ
    # น่าจะเป็นวัตถุ (foreground) ข้างนอกกรอบน่าจะเป็นพื้นหลัง
    mx = int(w * MARGIN_RATIO)
    my = int(h * MARGIN_RATIO)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)              # mask เปล่าที่ GrabCut จะเขียนผลลัพธ์ลงไป
    bgdModel = np.zeros((1, 65), np.float64)        # ตัวแปรภายในของ GrabCut (โมเดลสีพื้นหลัง)
    fgdModel = np.zeros((1, 65), np.float64)        # ตัวแปรภายในของ GrabCut (โมเดลสีวัตถุ)

    # เผื่อไว้กรณี GrabCut ทำงานพลาด: ให้ "ทั้งหมดในกรอบ rect" เป็น foreground ไปเลย
    fallback = np.zeros((h, w), np.uint8)
    fallback[my:h - my, mx:w - mx] = 255

    try:
        # รันอัลกอริทึม GrabCut จริง (8 iterations) โดยใช้ img (สี) + rect (กรอบเริ่มต้น) เป็น input
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
        # แปลผลลัพธ์จาก GrabCut (มี 4 ค่า: FGD, BGD, PR_FGD, PR_BGD) ให้เหลือ 0/255 แบบง่าย
        gt = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_ratio = gt.mean() / 255.0   # สัดส่วนพื้นที่ foreground ต่อภาพทั้งหมด (0.0-1.0)
        # GrabCut sometimes degenerates on images where the fruit fills almost
        # the whole frame (no real background to learn from). In that case its
        # foreground estimate becomes too small/too large to be usable, so we
        # fall back to "everything inside the rectangle is foreground".
        if fg_ratio < 0.15 or fg_ratio > 0.97:
            print(f"{f}: GrabCut degenerate (fg={fg_ratio:.2f}) -> fallback rect")
            gt = fallback   # ผลลัพธ์ GrabCut ดูผิดปกติเกินไป -> ใช้ fallback แทน
    except cv2.error as e:
        print("GrabCut failed on", f, "-> fallback to full rect", e)
        gt = fallback   # GrabCut error ตรงๆ (เช่น ภาพเสีย) -> ใช้ fallback แทนเช่นกัน

    # light cleanup: fill small holes, keep largest component
    # หา contour (เส้นขอบ) ทั้งหมดใน mask ที่ได้ แล้วเก็บไว้เฉพาะก้อนที่ใหญ่ที่สุด
    # (ตัด noise เล็กๆ ที่หลุดออกมาทิ้ง ก่อนเซฟเป็น ground truth จริง)
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)   # เลือก contour ที่มีพื้นที่มากสุด
        clean = np.zeros_like(gt)
        cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)  # ระบายก้อนนั้นให้เต็ม (เติมรู)
        gt = clean

    # เซฟผลลัพธ์ ground truth mask -> gt/<ชื่อไฟล์>_gt.png (0=พื้นหลัง, 255=ผลไม้)
    out_name = f.replace(".jpg", "_gt.png")
    cv2.imwrite(os.path.join(GT_DIR, out_name), gt)

print(f"Ground truth masks created for {len(files)} images -> {GT_DIR}/")

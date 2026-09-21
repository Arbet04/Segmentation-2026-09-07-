"""
Utility script (optional, NOT part of the main 01-06 pipeline)
Use this whenever you want to inspect a score map (scores/*_score.npy)
that was produced by 03_segment.py.

Usage:
    python3 view_score.py fruit_001
    python3 view_score.py fruit_023

It will:
  1. Print basic stats (shape, min, max, mean) to the terminal.
  2. Save a grayscale preview:  scores/fruit_001_score_view.png
  3. Save a color (heatmap) preview: scores/fruit_001_score_heatmap.png
"""
import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # backend แบบไม่เปิดหน้าต่าง (เซฟเป็นไฟล์ .png ตรงๆ)
import matplotlib.pyplot as plt

SCORE_DIR = "scores"   # โฟลเดอร์ที่เก็บไฟล์ *_score.npy (มาจาก output ของ 03_segment.py)

# รับ argument จาก command line เช่น "python3 view_score.py fruit_001"
# sys.argv[0] = ชื่อสคริปต์เอง, sys.argv[1] = ชื่อภาพที่ผู้ใช้พิมพ์ตามมา
if len(sys.argv) < 2:
    print("Usage: python3 view_score.py <image_base_name>")
    print("Example: python3 view_score.py fruit_001")
    sys.exit(1)   # ไม่ได้ระบุชื่อภาพมา -> แจ้งวิธีใช้แล้วออกจากโปรแกรม

base = sys.argv[1].replace(".jpg", "")   # ตัด .jpg ทิ้ง เผื่อผู้ใช้พิมพ์นามสกุลมาด้วย
score_path = os.path.join(SCORE_DIR, base + "_score.npy")   # เช่น scores/fruit_001_score.npy

if not os.path.exists(score_path):
    print(f"ไม่พบไฟล์: {score_path}")
    sys.exit(1)   # ไม่มีไฟล์นี้อยู่จริง (พิมพ์ชื่อผิด/ยังไม่ได้รัน 03_segment.py) -> ออกจากโปรแกรม

# --- นี่คือจุดที่ "ดึงข้อมูล" หลักของสคริปต์นี้ ---
# โหลดค่าตัวเลข score map (array 2 มิติ ค่า 0-255) ที่ 03_segment.py คำนวณและเซฟไว้ก่อนหน้า
score = np.load(score_path)

print(f"ไฟล์: {score_path}")
print(f"ขนาด (สูง, กว้าง): {score.shape}")
print(f"ชนิดข้อมูล: {score.dtype}")
print(f"ค่าต่ำสุด: {score.min()}")
print(f"ค่าสูงสุด: {score.max()}")
print(f"ค่าเฉลี่ย: {score.mean():.2f}")

# 1) grayscale preview
# แปลง array ตัวเลขให้เป็นภาพขาว-ดำโดยตรง (ค่ามาก=สว่าง, ค่าน้อย=มืด) แล้วเซฟเป็น .png
gray_out = os.path.join(SCORE_DIR, base + "_score_view.png")
cv2.imwrite(gray_out, score)
print(f"บันทึกภาพ grayscale -> {gray_out}")

# 2) color heatmap preview
# ใช้ matplotlib วาดภาพเดียวกันแต่ลงสี (cmap="jet") พร้อม colorbar บอกสเกลค่า ให้ดูง่ายกว่าขาวดำ
heat_out = os.path.join(SCORE_DIR, base + "_score_heatmap.png")
plt.figure(figsize=(6, 5))
plt.imshow(score, cmap="jet")
plt.colorbar(label="Color distance score (0-255)")
plt.title(f"Score map: {base}")
plt.tight_layout()
plt.savefig(heat_out, dpi=150)
plt.close()
print(f"บันทึกภาพ heatmap -> {heat_out}")

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORE_DIR = "scores"

if len(sys.argv) < 2:
    print("Usage: python3 view_score.py <image_base_name>")
    print("Example: python3 view_score.py fruit_001")
    sys.exit(1)

base = sys.argv[1].replace(".jpg", "")
score_path = os.path.join(SCORE_DIR, base + "_score.npy")

if not os.path.exists(score_path):
    print(f"ไม่พบไฟล์: {score_path}")
    sys.exit(1)

score = np.load(score_path)

print(f"ไฟล์: {score_path}")
print(f"ขนาด (สูง, กว้าง): {score.shape}")
print(f"ชนิดข้อมูล: {score.dtype}")
print(f"ค่าต่ำสุด: {score.min()}")
print(f"ค่าสูงสุด: {score.max()}")
print(f"ค่าเฉลี่ย: {score.mean():.2f}")

# 1) grayscale preview
gray_out = os.path.join(SCORE_DIR, base + "_score_view.png")
cv2.imwrite(gray_out, score)
print(f"บันทึกภาพ grayscale -> {gray_out}")

# 2) color heatmap preview
heat_out = os.path.join(SCORE_DIR, base + "_score_heatmap.png")
plt.figure(figsize=(6, 5))
plt.imshow(score, cmap="jet")
plt.colorbar(label="Color distance score (0-255)")
plt.title(f"Score map: {base}")
plt.tight_layout()
plt.savefig(heat_out, dpi=150)
plt.close()
print(f"บันทึกภาพ heatmap -> {heat_out}")

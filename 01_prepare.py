"""
Step 1: Prepare dataset
- Load all images from dataset50/
- Convert to RGB, resize so max dimension = 640 px (consistent + fast processing)
- Save as clean .jpg files into images/ with simple IDs (fruit_001.jpg ...)
"""
import os
from PIL import Image, ImageOps

# --- ตั้งค่าพื้นฐาน ---
SRC_DIR = "dataset50"   # โฟลเดอร์ input: ต้องมี dataset50/ อยู่ระดับเดียวกับไฟล์นี้ (relative path)
OUT_DIR = "images"      # โฟลเดอร์ output: ภาพที่ resize แล้วจะถูกเซฟไว้ที่นี่
MAX_DIM = 640           # ด้านที่ยาวที่สุดของภาพ จะถูกย่อไม่ให้เกินค่านี้ (px)

os.makedirs(OUT_DIR, exist_ok=True)  # สร้างโฟลเดอร์ images/ ถ้ายังไม่มี (ไม่ error ถ้ามีอยู่แล้ว)

# อ่านรายชื่อไฟล์ทั้งหมดใน dataset50/ แล้วเรียงลำดับตามชื่อ (A-Z)
# -> นี่คือจุดที่ "ดึงข้อมูลมาจากไหน": ดึงจากไฟล์จริงในโฟลเดอร์ dataset50/ บนดิสก์
files = sorted(os.listdir(SRC_DIR))
manifest = []  # เก็บคู่ (ชื่อไฟล์ใหม่, ชื่อไฟล์เดิม) ไว้เขียนลง manifest.csv ตอนท้าย

idx = 1  # ตัวนับสำหรับตั้งชื่อไฟล์ใหม่ fruit_001, fruit_002, ...
for f in files:
    path = os.path.join(SRC_DIR, f)  # เช่น "dataset50/Image_1.jpg"
    try:
        im = Image.open(path)               # เปิดไฟล์ภาพต้นฉบับ (อ่านจากดิสก์โดยตรง)
        im = ImageOps.exif_transpose(im)    # แก้ปัญหาภาพหมุนผิดทิศ (อ่าน metadata EXIF ในไฟล์ภาพเอง)
        im = im.convert("RGB")               # บังคับให้เป็น RGB เสมอ (กัน mode แปลกๆ เช่น CMYK, PNG โปร่งใส, .gif)
    except Exception as e:
        print("SKIP (cannot open):", f, e)  # ถ้าเปิดไฟล์ไม่ได้ (ไฟล์เสีย/นามสกุลแปลก) ให้ข้ามไปเลย ไม่หยุดทั้งโปรแกรม
        continue

    w, h = im.size                    # ขนาดภาพเดิม (กว้าง, สูง) หน่วย pixel
    scale = MAX_DIM / max(w, h)       # คำนวณอัตราส่วนย่อ จากด้านที่ยาวที่สุด
    if scale < 1:                     # ย่อเฉพาะภาพที่ใหญ่กว่า MAX_DIM เท่านั้น (ภาพเล็กกว่าจะไม่ขยาย)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)  # resize ด้วย LANCZOS (คุณภาพดีสุด)

    out_name = f"fruit_{idx:03d}.jpg"                       # ตั้งชื่อใหม่ เช่น fruit_001.jpg
    im.save(os.path.join(OUT_DIR, out_name), quality=92)    # เซฟไฟล์ผลลัพธ์ลง images/
    manifest.append((out_name, f))                          # จำคู่ชื่อใหม่-เก่าไว้
    idx += 1

print(f"Prepared {len(manifest)} images -> {OUT_DIR}/")

# เขียนตารางเทียบชื่อไฟล์ใหม่ <-> ชื่อไฟล์ต้นฉบับ ลงไฟล์ results/manifest.csv
# ⚠️ จุดนี้ "ดึง/เขียน" ไปยังโฟลเดอร์ results/ ซึ่งสคริปต์นี้ไม่ได้สร้างให้อัตโนมัติ
#    (ไม่มี os.makedirs("results") ด้านบน) ถ้าไม่มีโฟลเดอร์ results/ อยู่ก่อน
#    บรรทัดนี้จะทำให้เกิด FileNotFoundError
#
with open("results/manifest.csv", "w", encoding="utf-8") as fp:
    fp.write("new_name,original_name\n")
    for a, b in manifest:
        fp.write(f"{a},{b}\n")

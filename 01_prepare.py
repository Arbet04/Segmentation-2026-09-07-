"""
Step 1: Prepare dataset
- Load all images from dataset50/
- Convert to RGB, resize so max dimension = 640 px (consistent + fast processing)
- Save as clean .jpg files into images/ with simple IDs (fruit_001.jpg ...)
"""
import os
from PIL import Image, ImageOps

SRC_DIR = "dataset50"
OUT_DIR = "images"
MAX_DIM = 640

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(os.listdir(SRC_DIR))
manifest = []

idx = 1
for f in files:
    path = os.path.join(SRC_DIR, f)
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)  # fix rotation
        im = im.convert("RGB")
    except Exception as e:
        print("SKIP (cannot open):", f, e)
        continue

    w, h = im.size
    scale = MAX_DIM / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    out_name = f"fruit_{idx:03d}.jpg"
    im.save(os.path.join(OUT_DIR, out_name), quality=92)
    manifest.append((out_name, f))
    idx += 1

print(f"Prepared {len(manifest)} images -> {OUT_DIR}/")
with open("results/manifest.csv", "w", encoding="utf-8") as fp:
    fp.write("new_name,original_name\n")
    for a, b in manifest:
        fp.write(f"{a},{b}\n")

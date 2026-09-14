# Fruit Segmentation Workshop — Confusion Matrix & Morphology Evaluation

โปรเจกต์นี้ทำตามโจทย์ Workshop (แยกวัตถุ/Object ออกจากพื้นหลัง/Background ด้วย
Morphological Image Processing แล้วประเมินผลด้วย Confusion Matrix และ ROC Curve)

**Positive class = ผลไม้ (fruit) | Negative class = พื้นหลัง (background)**

## โครงสร้างไฟล์

```
01_prepare.py         Step 2: เตรียม dataset (resize เป็นขนาดมาตรฐาน, แก้ EXIF rotation)
02_ground_truth.py    Step 2: สร้าง Ground Truth ด้วย GrabCut (กึ่งอัตโนมัติ)
03_segment.py         Step 3: Algorithm segmentation หลัก
                       (color-distance ใน Lab space จากสีพื้นหลังที่ขอบภาพ
                        + Otsu thresholding) -> ได้ raw mask
04_morphology.py       Step 4: ปรับปรุงผลด้วย Morphology
                       (Opening -> Closing -> ลบ component เล็ก/noise -> เติมรู)
05_evaluate.py         Step 5: Evaluation
                       - Confusion Matrix (pooled TP/FP/FN/TN ทั้ง 50 ภาพ)
                       - Accuracy / Precision / Recall / F1 (raw vs after morphology)
                       - ROC Curve (sweep threshold บน score map) + AUC

images/         ภาพ dataset ที่ resize แล้ว (50 ภาพ, max dim 640px)
gt/             Ground truth mask (<name>_gt.png, 0/255)
pred_raw/       Mask ก่อน morphology (<name>_raw.png)
pred_morph/     Mask หลัง morphology (<name>_morph.png)
viz/            ภาพผลลัพธ์สำหรับนำเสนอ (confusion_matrix.png, roc_curve.png,
                 metrics_bar.png, examples_grid.png)
results/        ตัวเลขผลลัพธ์ (metrics.json, per_image_metrics.csv, roc.json,
                 manifest.csv)
```

## วิธีรันซ้ำ (Reproduce)

ต้องมีโฟลเดอร์ `dataset50/` (ภาพต้นฉบับ) วางไว้ระดับเดียวกับสคริปต์ แล้วรันตามลำดับ:

```bash
python3 01_prepare.py        # -> images/
python3 02_ground_truth.py   # -> gt/
python3 03_segment.py        # -> scores/, pred_raw/
python3 04_morphology.py     # -> pred_morph/
python3 05_evaluate.py       # -> results/*.json, results/*.csv
```

ไลบรารีที่ใช้: `opencv-python`, `numpy`, `matplotlib`, `Pillow`, `scipy`
(ติดตั้งด้วย `pip install opencv-python numpy matplotlib pillow scipy`)

## สรุปผลลัพธ์หลัก (pooled, 50 ภาพ)

| Metric      | Before Morphology | After Morphology |
|-------------|-------------------|-------------------|
| Accuracy    | 0.626              | 0.583              |
| Precision   | 0.519              | 0.471              |
| Recall      | 0.470              | 0.635              |
| F1-score    | 0.493              | **0.541**          |

ROC AUC (sweep threshold บน score map) = **0.630**

รายละเอียดตัวเลขแบบเต็มดูได้ที่ `results/metrics.json`,
`results/per_image_metrics.csv` และ `results/roc.json`

## หมายเหตุเรื่อง Ground Truth

Dataset ไม่มี ground truth มาให้ จึงสร้างขึ้นเองด้วย GrabCut โดยกำหนดกรอบ
(rectangle) ครอบพื้นที่กลางภาพเป็น seed แล้วให้ GrabCut แยกวัตถุออกจากพื้นหลัง
โดยอัตโนมัติ — เป็น ground truth แบบประมาณ (approximate/semi-automatic)
ไม่ใช่การ label มือทีละพิกเซล จึงอาจมีความคลาดเคลื่อนในบางภาพ (ดูตัวอย่างใน
`viz/examples_grid.png`)

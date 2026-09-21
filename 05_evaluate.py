"""
Step 5: Evaluate experimental results using the Confusion Matrix
(directly follows the "Evaluating Experimental Results" slide deck)

- Positive class (1) = fruit (object)
- Negative class (0) = background

For each image we compare the predicted mask against the ground-truth
mask pixel-by-pixel and accumulate TP / FP / TN / FN over ALL images
(a pooled / micro-averaged confusion matrix), for:
    (a) RAW prediction   (Otsu threshold only, before morphology)
    (b) MORPH prediction (after Opening -> Closing -> clean-up)

From the pooled confusion matrix we compute:
    Accuracy, Precision, Recall (TPR), Specificity (TNR),
    False Positive Rate, F1-score

We also sweep the threshold on the continuous score map (Step 3) to
build a pooled ROC curve (TPR vs FPR at many thresholds) and compute
the Area Under the Curve (AUC).
"""
import os
import cv2
import numpy as np
import json

# --- ตั้งค่าพื้นฐาน: ดึงข้อมูลจาก output ของทุกไฟล์ก่อนหน้า ---
IMG_DIR = "images"      # ใช้แค่หารายชื่อไฟล์ที่ต้องวนประมวลผล (output ของ 01_prepare.py)
GT_DIR = "gt"            # เฉลย (ground truth) -> output ของ 02_ground_truth.py
RAW_DIR = "pred_raw"     # คำตอบก่อน morphology -> output ของ 03_segment.py
MORPH_DIR = "pred_morph" # คำตอบหลัง morphology -> output ของ 04_morphology.py
SCORE_DIR = "scores"     # score map แบบต่อเนื่อง (.npy) -> output ของ 03_segment.py (ใช้ทำ ROC)
RES_DIR = "results"      # ผลลัพธ์ตัวเลขทั้งหมดของไฟล์นี้จะเซฟไว้ที่นี่
os.makedirs(RES_DIR, exist_ok=True)

# ดึงรายชื่อไฟล์ทั้งหมดจาก images/ แล้วใช้ชื่อเดียวกัน (base name) ไปหาไฟล์ที่เกี่ยวข้องในโฟลเดอร์อื่น
files = sorted(os.listdir(IMG_DIR))


def confusion_counts(pred_bin, gt_bin):
    """pred_bin, gt_bin: boolean arrays, same shape. Positive = True (fruit)."""
    # นับพิกเซลทีละแบบ โดยเทียบ "คำตอบที่ทำนาย" (pred_bin) กับ "เฉลย" (gt_bin)
    tp = int(np.sum(pred_bin & gt_bin))    # ทำนายว่าใช่ + เฉลยก็ใช่ -> True Positive
    fp = int(np.sum(pred_bin & ~gt_bin))   # ทำนายว่าใช่ + เฉลยไม่ใช่ -> False Positive
    fn = int(np.sum(~pred_bin & gt_bin))   # ทำนายว่าไม่ใช่ + เฉลยใช่ -> False Negative
    tn = int(np.sum(~pred_bin & ~gt_bin))  # ทำนายว่าไม่ใช่ + เฉลยก็ไม่ใช่ -> True Negative
    return tp, fp, fn, tn


def to_native(d):
    """Recursively convert numpy scalar types to native Python types for JSON."""
    # json.dump() เขียนชนิดข้อมูลของ numpy (np.int64, np.float64) ไม่ได้โดยตรง
    # ฟังก์ชันนี้แปลงค่าพวกนั้นให้เป็น int/float ธรรมดาของ Python ก่อนเซฟ
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def metrics_from_counts(tp, fp, fn, tn):
    # รับค่า TP/FP/FN/TN (จาก confusion_counts) มาแปลงเป็นตัวชี้วัดมาตรฐานทั้งหมด
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0  # TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Total=total,
                Accuracy=acc, Precision=precision, Recall=recall,
                Specificity=specificity, FPR=fpr, F1=f1)


# ---------- Pooled confusion matrix: RAW vs MORPH ----------
# agg_raw / agg_morph เก็บผลรวม TP/FP/FN/TN สะสมจาก "ทุกภาพ" (ไม่แยกรายภาพ)
agg_raw = np.zeros(4, dtype=np.int64)   # tp fp fn tn
agg_morph = np.zeros(4, dtype=np.int64)

per_image_rows = []  # เก็บผลลัพธ์แยกรายภาพ ไว้เขียนลง per_image_metrics.csv ทีหลัง

for f in files:
    base = f.replace(".jpg", "")
    # อ่าน mask 3 ตัวของภาพเดียวกัน จากคนละโฟลเดอร์ (ใช้ชื่อไฟล์ base เดียวกันเชื่อมกัน)
    # ">127" คือแปลง pixel ค่า 0-255 ให้เป็น True/False (255=วัตถุ, 0=พื้นหลัง)
    gt = cv2.imread(os.path.join(GT_DIR, base + "_gt.png"), cv2.IMREAD_GRAYSCALE) > 127
    raw = cv2.imread(os.path.join(RAW_DIR, base + "_raw.png"), cv2.IMREAD_GRAYSCALE) > 127
    morph = cv2.imread(os.path.join(MORPH_DIR, base + "_morph.png"), cv2.IMREAD_GRAYSCALE) > 127

    r = confusion_counts(raw, gt)      # เทียบ raw กับ gt ของภาพนี้
    m = confusion_counts(morph, gt)    # เทียบ morph กับ gt ของภาพนี้
    agg_raw += np.array(r)              # สะสมเข้าผลรวมทั้ง dataset
    agg_morph += np.array(m)

    mr = metrics_from_counts(*r)        # แปลงเป็น metrics เฉพาะภาพนี้ (raw)
    mm = metrics_from_counts(*m)        # แปลงเป็น metrics เฉพาะภาพนี้ (morph)
    per_image_rows.append({
        "image": base,
        "raw_acc": mr["Accuracy"], "raw_f1": mr["F1"],
        "morph_acc": mm["Accuracy"], "morph_f1": mm["F1"],
    })

# แปลงผลรวมสะสมทั้ง dataset เป็น metrics สุดท้าย (นี่คือตัวเลข "pooled" ที่ print/เซฟ)
raw_metrics = metrics_from_counts(*agg_raw)
morph_metrics = metrics_from_counts(*agg_morph)

print("=== Pooled Confusion Matrix : RAW (before morphology) ===")
print(f"TP={raw_metrics['TP']}  FP={raw_metrics['FP']}  FN={raw_metrics['FN']}  TN={raw_metrics['TN']}")
print(f"Accuracy={raw_metrics['Accuracy']:.4f}  Precision={raw_metrics['Precision']:.4f}  "
      f"Recall={raw_metrics['Recall']:.4f}  F1={raw_metrics['F1']:.4f}")

print("\n=== Pooled Confusion Matrix : AFTER Morphology (Open->Close->CC->fill) ===")
print(f"TP={morph_metrics['TP']}  FP={morph_metrics['FP']}  FN={morph_metrics['FN']}  TN={morph_metrics['TN']}")
print(f"Accuracy={morph_metrics['Accuracy']:.4f}  Precision={morph_metrics['Precision']:.4f}  "
      f"Recall={morph_metrics['Recall']:.4f}  F1={morph_metrics['F1']:.4f}")

# เซฟผลลัพธ์รวม (raw vs morph) -> results/metrics.json
with open(os.path.join(RES_DIR, "metrics.json"), "w") as fp:
    json.dump({"raw": to_native(raw_metrics), "morph": to_native(morph_metrics)}, fp, indent=2)

# เซฟผลลัพธ์แยกรายภาพ -> results/per_image_metrics.csv
with open(os.path.join(RES_DIR, "per_image_metrics.csv"), "w", encoding="utf-8") as fp:
    fp.write("image,raw_acc,raw_f1,morph_acc,morph_f1\n")
    for row in per_image_rows:
        fp.write(f"{row['image']},{row['raw_acc']:.4f},{row['raw_f1']:.4f},"
                  f"{row['morph_acc']:.4f},{row['morph_f1']:.4f}\n")

# ---------- ROC curve (pooled, sweeping threshold on score map) ----------
# ส่วนนี้ไม่ใช้ pred_raw/pred_morph (ที่ตัดสินใจแล้วด้วย threshold เดียว)
# แต่ดึง score map ดิบ (scores/*.npy) มาลองตัด threshold หลายค่า เพื่อดูว่า
# ถ้าเปลี่ยนจุดตัดสินใจ ผล TPR/FPR จะเปลี่ยนไปยังไง (สร้างเป็นกราฟ ROC)
thresholds = list(range(0, 256, 5))   # ลอง threshold ตั้งแต่ 0 ถึง 255 ทีละ 5
tpr_list, fpr_list = [], []

# Pre-load all score maps & gts once (downsample for speed)
scores_all = []
gts_all = []
for f in files:
    base = f.replace(".jpg", "")
    score = np.load(os.path.join(SCORE_DIR, base + "_score.npy"))   # โหลด score map ดิบจาก 03_segment.py
    gt = cv2.imread(os.path.join(GT_DIR, base + "_gt.png"), cv2.IMREAD_GRAYSCALE) > 127
    # downsample for speed (ROC over full-res pooled pixels of 50 images is heavy)
    # ย่อภาพทุกภาพให้เหลือ 120x90 ก่อน เพื่อให้ loop threshold 256/5=52 รอบ ทำงานเร็วขึ้นมาก
    score_small = cv2.resize(score, (120, 90), interpolation=cv2.INTER_AREA)
    gt_small = cv2.resize(gt.astype(np.uint8) * 255, (120, 90), interpolation=cv2.INTER_NEAREST) > 127
    scores_all.append(score_small)
    gts_all.append(gt_small)

scores_all = np.stack(scores_all)  # (N,90,120)  รวมทุกภาพเป็น array เดียว 3 มิติ (N=จำนวนภาพ)
gts_all = np.stack(gts_all)

# วนตัด threshold แต่ละค่า -> ได้ mask ชั่วคราวของทุกภาพพร้อมกัน -> นับ TPR/FPR รวมทั้ง dataset
for t in thresholds:
    pred = scores_all > t
    tp = np.sum(pred & gts_all)
    fp = np.sum(pred & ~gts_all)
    fn = np.sum(~pred & gts_all)
    tn = np.sum(~pred & ~gts_all)
    tpr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# sort by fpr for correct AUC integration (trapezoid), curve naturally monotonic-ish
order = np.argsort(fpr_list)   # เรียงจุดตาม FPR จากน้อยไปมาก (จำเป็นก่อนคำนวณพื้นที่ใต้กราฟ)
fpr_sorted = np.array(fpr_list)[order]
tpr_sorted = np.array(tpr_list)[order]
auc = np.trapezoid(tpr_sorted, fpr_sorted)   # คำนวณพื้นที่ใต้กราฟ ROC (Area Under Curve)

print(f"\nROC AUC (pooled, sweeping Otsu-score threshold) = {auc:.4f}")

# เซฟผลลัพธ์ ROC (threshold แต่ละค่า, TPR, FPR, AUC รวม) -> results/roc.json
with open(os.path.join(RES_DIR, "roc.json"), "w") as fp:
    json.dump({
        "thresholds": [int(t) for t in thresholds],
        "tpr": [float(x) for x in tpr_list],
        "fpr": [float(x) for x in fpr_list],
        "auc": float(auc),
    }, fp, indent=2)

print("\nSaved: results/metrics.json, results/per_image_metrics.csv, results/roc.json")

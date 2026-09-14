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

IMG_DIR = "images"
GT_DIR = "gt"
RAW_DIR = "pred_raw"
MORPH_DIR = "pred_morph"
SCORE_DIR = "scores"
RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)

files = sorted(os.listdir(IMG_DIR))


def confusion_counts(pred_bin, gt_bin):
    """pred_bin, gt_bin: boolean arrays, same shape. Positive = True (fruit)."""
    tp = int(np.sum(pred_bin & gt_bin))
    fp = int(np.sum(pred_bin & ~gt_bin))
    fn = int(np.sum(~pred_bin & gt_bin))
    tn = int(np.sum(~pred_bin & ~gt_bin))
    return tp, fp, fn, tn


def to_native(d):
    """Recursively convert numpy scalar types to native Python types for JSON."""
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
agg_raw = np.zeros(4, dtype=np.int64)   # tp fp fn tn
agg_morph = np.zeros(4, dtype=np.int64)

per_image_rows = []

for f in files:
    base = f.replace(".jpg", "")
    gt = cv2.imread(os.path.join(GT_DIR, base + "_gt.png"), cv2.IMREAD_GRAYSCALE) > 127
    raw = cv2.imread(os.path.join(RAW_DIR, base + "_raw.png"), cv2.IMREAD_GRAYSCALE) > 127
    morph = cv2.imread(os.path.join(MORPH_DIR, base + "_morph.png"), cv2.IMREAD_GRAYSCALE) > 127

    r = confusion_counts(raw, gt)
    m = confusion_counts(morph, gt)
    agg_raw += np.array(r)
    agg_morph += np.array(m)

    mr = metrics_from_counts(*r)
    mm = metrics_from_counts(*m)
    per_image_rows.append({
        "image": base,
        "raw_acc": mr["Accuracy"], "raw_f1": mr["F1"],
        "morph_acc": mm["Accuracy"], "morph_f1": mm["F1"],
    })

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

with open(os.path.join(RES_DIR, "metrics.json"), "w") as fp:
    json.dump({"raw": to_native(raw_metrics), "morph": to_native(morph_metrics)}, fp, indent=2)

with open(os.path.join(RES_DIR, "per_image_metrics.csv"), "w", encoding="utf-8") as fp:
    fp.write("image,raw_acc,raw_f1,morph_acc,morph_f1\n")
    for row in per_image_rows:
        fp.write(f"{row['image']},{row['raw_acc']:.4f},{row['raw_f1']:.4f},"
                  f"{row['morph_acc']:.4f},{row['morph_f1']:.4f}\n")

# ---------- ROC curve (pooled, sweeping threshold on score map) ----------
thresholds = list(range(0, 256, 5))
tpr_list, fpr_list = [], []

# Pre-load all score maps & gts once (downsample for speed)
scores_all = []
gts_all = []
for f in files:
    base = f.replace(".jpg", "")
    score = np.load(os.path.join(SCORE_DIR, base + "_score.npy"))
    gt = cv2.imread(os.path.join(GT_DIR, base + "_gt.png"), cv2.IMREAD_GRAYSCALE) > 127
    # downsample for speed (ROC over full-res pooled pixels of 50 images is heavy)
    score_small = cv2.resize(score, (120, 90), interpolation=cv2.INTER_AREA)
    gt_small = cv2.resize(gt.astype(np.uint8) * 255, (120, 90), interpolation=cv2.INTER_NEAREST) > 127
    scores_all.append(score_small)
    gts_all.append(gt_small)

scores_all = np.stack(scores_all)  # (N,90,120)
gts_all = np.stack(gts_all)

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
order = np.argsort(fpr_list)
fpr_sorted = np.array(fpr_list)[order]
tpr_sorted = np.array(tpr_list)[order]
auc = np.trapezoid(tpr_sorted, fpr_sorted)

print(f"\nROC AUC (pooled, sweeping Otsu-score threshold) = {auc:.4f}")

with open(os.path.join(RES_DIR, "roc.json"), "w") as fp:
    json.dump({
        "thresholds": [int(t) for t in thresholds],
        "tpr": [float(x) for x in tpr_list],
        "fpr": [float(x) for x in fpr_list],
        "auc": float(auc),
    }, fp, indent=2)

print("\nSaved: results/metrics.json, results/per_image_metrics.csv, results/roc.json")

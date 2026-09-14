"""
Step 6 (bonus): Generate all visualizations in viz/
(This script was originally run ad-hoc and is now saved properly so the
 whole pipeline, including viz/, can be reproduced end-to-end.)

Requires: results/metrics.json, results/roc.json, results/per_image_metrics.csv
          and images/, gt/, pred_raw/, pred_morph/  (i.e. run 01-05 first)

Outputs (viz/):
  confusion_matrix.png  - heatmap of TP/FP/FN/TN, before vs after morphology
  roc_curve.png          - ROC curve + AUC
  metrics_bar.png        - bar chart of Accuracy/Precision/Recall/F1
  examples_grid.png      - 3 example images (best / moderate / worst F1)
                            x 4 columns (Original | GT | Raw | Morph)
"""
import os
import csv
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = "results"
VIZ_DIR = "viz"
os.makedirs(VIZ_DIR, exist_ok=True)

metrics = json.load(open(os.path.join(RES_DIR, "metrics.json")))
roc = json.load(open(os.path.join(RES_DIR, "roc.json")))
per_image = list(csv.DictReader(open(os.path.join(RES_DIR, "per_image_metrics.csv"))))

# ---------------------------------------------------------------- #
# 1) Confusion matrix heatmap (before vs after morphology)
# ---------------------------------------------------------------- #
def plot_cm(ax, d, title):
    cm = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]])
    cm_pct = cm / cm.sum() * 100
    ax.imshow(cm_pct, cmap="Greens", vmin=0, vmax=50)
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                     ha="center", va="center", fontsize=11,
                     color="white" if cm_pct[i, j] > 25 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nPositive", "Actual\nNegative"])
    ax.set_title(title, fontsize=13)


fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_cm(axes[0], metrics["raw"],
        f"Before Morphology\nAcc={metrics['raw']['Accuracy']:.3f}  F1={metrics['raw']['F1']:.3f}")
plot_cm(axes[1], metrics["morph"],
        f"After Morphology\nAcc={metrics['morph']['Accuracy']:.3f}  F1={metrics['morph']['F1']:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------- #
# 2) ROC curve
# ---------------------------------------------------------------- #
fpr = np.array(roc["fpr"]); tpr = np.array(roc["tpr"]); auc = roc["auc"]
order = np.argsort(fpr)
fpr_s, tpr_s = fpr[order], tpr[order]

plt.figure(figsize=(6, 6))
plt.plot(fpr_s, tpr_s, color="#2E86AB", linewidth=2.5, label=f"Fruit Segmentation (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Fruit (Positive) vs Background (Negative)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim(0, 1); plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "roc_curve.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------- #
# 3) Metrics bar chart (Accuracy/Precision/Recall/F1, before vs after)
# ---------------------------------------------------------------- #
metric_names = ["Accuracy", "Precision", "Recall", "F1"]
raw_vals = [metrics["raw"][k] for k in metric_names]
morph_vals = [metrics["morph"][k] for k in metric_names]

x = np.arange(len(metric_names))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - w / 2, raw_vals, w, label="Before Morphology", color="#F18F01")
b2 = ax.bar(x + w / 2, morph_vals, w, label="After Morphology", color="#2E86AB")
ax.set_xticks(x); ax.set_xticklabels(metric_names)
ax.set_ylim(0, 0.8)
ax.set_ylabel("Score")
ax.set_title("Segmentation Performance: Before vs After Morphology")
ax.legend()
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "metrics_bar.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------- #
# 4) Example grid: best / moderate / worst F1 images
# ---------------------------------------------------------------- #
rows_sorted = sorted(per_image, key=lambda r: float(r["morph_f1"]), reverse=True)
best = rows_sorted[0]
worst = rows_sorted[-1]
mid = rows_sorted[len(rows_sorted) // 2]

examples = [
    (best["image"], f"Best case (F1={float(best['morph_f1']):.2f})"),
    (mid["image"], f"Moderate case (F1={float(mid['morph_f1']):.2f})"),
    (worst["image"], f"Difficult case (F1={float(worst['morph_f1']):.2f})"),
]
cols = ["Original", "Ground Truth", "Raw (Otsu)", "After Morphology"]

fig, axes = plt.subplots(len(examples), 4, figsize=(12, 9))
for i, (base, label) in enumerate(examples):
    img = cv2.cvtColor(cv2.imread(f"images/{base}.jpg"), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(f"gt/{base}_gt.png", 0)
    raw = cv2.imread(f"pred_raw/{base}_raw.png", 0)
    morph = cv2.imread(f"pred_morph/{base}_morph.png", 0)
    imgs = [img, gt, raw, morph]
    cmaps = [None, "gray", "gray", "gray"]
    for j in range(4):
        ax = axes[i, j]
        ax.imshow(imgs[j], cmap=cmaps[j])
        ax.axis("off")
        if i == 0:
            ax.set_title(cols[j], fontsize=12)
    axes[i, 0].text(-0.15, 0.5, label, transform=axes[i, 0].transAxes, rotation=90,
                      va="center", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "examples_grid.png"), dpi=150)
plt.close()

print(f"Saved 4 visualization files -> {VIZ_DIR}/")

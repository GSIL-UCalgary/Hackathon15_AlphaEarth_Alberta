"""
Calculate Classification Metrics using the CLIPPED labels TIF.
Remaps Canada-wide IDs → Alberta IDs (0-12) then compares with ground truth.

Three conditions enforced before any metric is computed:
  1. Spatial size of GT and classified TIF must be identical.
  2. The valid-pixel mask is derived exclusively from the GT (-99 = background).
     That same mask is applied to the classified TIF — GT is the sole authority
     on which pixels are inside Alberta.
  3. Pixel values must be in [0-12].
       • GT is already in [0-12] (verified at load time).
       • Classified TIF is in Canada-wide IDs → remapped to [0-12] via
         CANADA_TO_ALBERTA before the mask is applied.

Usage:
    python calculate_metrics_tif.py --model_name BasicUNet --sensor_name landsat8
    python calculate_metrics_tif.py --model_name BasicUNet --sensor_name landsat8 \
        --tif_path /explicit/path/to/labels_CLIPPED.tif
"""

import numpy as np
import rasterio
from pathlib import Path
import json
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ==================== CONSTANTS ====================

NUM_CLASSES  = 13
IGNORE_VALUE = -99

ALBERTA_CLASSES = {
    0:  "Temperate needleleaf forest",
    1:  "Sub-polar taiga forest",
    2:  "Temperate broadleaf forest",
    3:  "Mixed forest",
    4:  "Temperate shrubland",
    5:  "Temperate grassland",
    6:  "Polar grassland-lichen",
    7:  "Wetland",
    8:  "Cropland",
    9:  "Barren lands",
    10: "Urban",
    11: "Water",
    12: "Snow/ice",
}

# Alberta (0-12) → Canada-wide IDs  (how the TIF was saved)
ALBERTA_TO_CANADA = {
    0: 1,  1: 2,  2: 5,  3: 6,  4: 8,  5: 10,
    6: 12, 7: 14, 8: 15, 9: 16, 10: 17, 11: 18, 12: 19,
}
# Reverse: Canada-wide ID → Alberta (0-12)
CANADA_TO_ALBERTA = {v: k for k, v in ALBERTA_TO_CANADA.items()}


# ==================== LOADING & VALIDATION ====================

def load_ground_truth(path: str):
    """
    Load GT GeoTIFF.  Values must be 0-12 (Alberta classes) or -99 (background).
    Enforces Condition 3 for the GT side.
    """
    print(f"\n📂  Ground truth : {path}")
    with rasterio.open(path) as src:
        gt = src.read(1).astype(np.int16)
        # Replace any rasterio-declared nodata with our canonical IGNORE_VALUE
        if src.nodata is not None and int(src.nodata) != IGNORE_VALUE:
            gt[gt == int(src.nodata)] = IGNORE_VALUE
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    # ── Condition 3 (GT side): verify all non-background values are in [0-12] ─
    unique_vals   = np.unique(gt)
    unexpected    = [int(v) for v in unique_vals if v != IGNORE_VALUE and not (0 <= v <= 12)]
    if unexpected:
        raise ValueError(
            f"GT contains values outside [0-12]: {unexpected}. "
            "Check the remapping of the ground truth file."
        )

    valid = gt != IGNORE_VALUE
    print(f"   Shape             : {gt.shape}")
    print(f"   Valid pixels      : {valid.sum():,} / {gt.size:,}  ({valid.mean()*100:.2f}%)")
    print(f"   Unique (valid)    : {np.unique(gt[valid]).tolist()}")
    print(f"   ✅ Condition 3 (GT): all values confirmed in [0-12]")
    return gt, transform, crs, bounds


def load_and_remap_tif(path: str):
    """
    Load classified TIF (Canada-wide IDs) and remap to Alberta [0-12].
    Enforces Condition 3 for the prediction side.
    """
    print(f"\n📂  Classified TIF : {path}")
    with rasterio.open(path) as src:
        raw       = src.read(1).astype(np.int16)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    print(f"   Shape (raw)       : {raw.shape}")
    print(f"   Unique (raw)      : {np.unique(raw).tolist()}")

    # ── Condition 3 (TIF side): remap Canada IDs → Alberta [0-12] ─────────────
    # Initialise everything as IGNORE_VALUE; only overwrite known Canada IDs.
    # This means TIF value 0 (background fill from np.zeros_like in classify_scene.py)
    # and any other unmapped value stays as IGNORE_VALUE automatically.
    remapped = np.full(raw.shape, IGNORE_VALUE, dtype=np.int16)
    for canada_id, alberta_id in CANADA_TO_ALBERTA.items():
        remapped[raw == canada_id] = alberta_id

    valid = remapped != IGNORE_VALUE
    print(f"   Unique (remapped) : {np.unique(remapped[valid]).tolist()}")
    print(f"   Valid (remapped)  : {valid.sum():,} / {remapped.size:,}  ({valid.mean()*100:.2f}%)")
    print(f"   ✅ Condition 3 (TIF): remapped to [0-12]; unmapped values set to -99")
    return remapped, transform, crs, bounds


def check_alignment(gt_transform, gt_crs, gt_bounds,
                    pred_transform, pred_crs, pred_bounds, label="Prediction"):
    print(f"\n🔍  Alignment check — {label} vs Ground Truth")
    crs_ok       = gt_crs       == pred_crs
    transform_ok = gt_transform == pred_transform
    bounds_ok    = gt_bounds    == pred_bounds
    print(f"   CRS match       : {'✅' if crs_ok       else '❌'}  {pred_crs if not crs_ok else ''}")
    print(f"   Transform match : {'✅' if transform_ok else '❌'}")
    print(f"   Bounds match    : {'✅' if bounds_ok    else '❌'}")
    if not (crs_ok and transform_ok and bounds_ok):
        print("   ⚠️  Misalignment detected — metrics may be unreliable!")
    else:
        print("   ✅  Perfect alignment")


# ==================== METRICS ====================

def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    """
    Compute all classification metrics with all three conditions enforced.

    Condition 1: shape equality checked before this function is called.
    Condition 2: valid mask built from GT only; same mask applied to pred.
    Condition 3: GT and pred are already in [0-12] / IGNORE_VALUE on entry.
    """

    # ── Condition 1: shape must match ────────────────────────────────────────
    if gt.shape != pred.shape:
        raise ValueError(
            f"❌ Condition 1 FAILED — shape mismatch: GT={gt.shape} vs PRED={pred.shape}"
        )
    print(f"\n   ✅ Condition 1: shapes match {gt.shape}")

    # ── Condition 2: valid mask from GT only ──────────────────────────────────
    # A pixel is valid if and only if the GT says it is inside Alberta (0-12).
    # The pred value at that location may still be IGNORE_VALUE (unmapped TIF
    # pixel); those 21k pixels are excluded from metrics but reported.
    valid_mask = (gt != IGNORE_VALUE) & (gt >= 0) & (gt < NUM_CLASSES)

    gt_valid   = gt[valid_mask]              # 1-D, dtype int16, values 0-12
    pred_valid = pred[valid_mask]            # 1-D, dtype int16, values 0-12 or -99

    n_gt_valid      = int(valid_mask.sum())
    n_pred_unmapped = int((pred_valid == IGNORE_VALUE).sum())
    n_compared      = n_gt_valid - n_pred_unmapped

    print(f"   ✅ Condition 2: valid mask from GT → {n_gt_valid:,} pixels inside Alberta")
    if n_pred_unmapped > 0:
        print(f"   ⚠️  {n_pred_unmapped:,} GT-valid pixels have no TIF prediction "
              f"({n_pred_unmapped/n_gt_valid*100:.4f}%) — excluded from metrics")

    # Final arrays used for metric computation: both must be in [0-12]
    comparison_mask = valid_mask & (pred != IGNORE_VALUE) & (pred >= 0) & (pred < NUM_CLASSES)
    y_true = gt[comparison_mask].astype(np.int32)
    y_pred = pred[comparison_mask].astype(np.int32)
    n      = len(y_true)

    print(f"   ✅ Condition 3: values in [0-12] confirmed; computing metrics on {n:,} pixels")

    if n == 0:
        raise RuntimeError("No valid pixels remain after applying all conditions.")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    tp      = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)   # GT support per class
    col_sum = cm.sum(axis=0).astype(np.float64)   # predicted count per class
    support = row_sum
    present = np.where(support > 0)[0]            # classes present in GT

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class = {}
    for i in range(NUM_CLASSES):
        if support[i] == 0:
            per_class[i] = {
                "class_name": ALBERTA_CLASSES[i], "support": 0,
                "accuracy": None, "precision": None,
                "recall": None,   "f1_score": None, "iou": None,
                "tp": 0, "fp": 0, "fn": 0,
            }
            continue

        recall_i    = tp[i] / support[i]                               # = per-class accuracy
        precision_i = tp[i] / col_sum[i] if col_sum[i] > 0 else 0.0
        f1_denom    = precision_i + recall_i
        f1_i        = 2 * precision_i * recall_i / f1_denom if f1_denom > 0 else 0.0
        iou_denom   = support[i] + col_sum[i] - tp[i]
        iou_i       = tp[i] / iou_denom if iou_denom > 0 else 0.0

        per_class[i] = {
            "class_name": ALBERTA_CLASSES[i],
            "support":    int(support[i]),
            "accuracy":   float(recall_i),
            "precision":  float(precision_i),
            "recall":     float(recall_i),
            "f1_score":   float(f1_i),
            "iou":        float(iou_i),
            "tp":         int(tp[i]),
            "fp":         int(col_sum[i] - tp[i]),
            "fn":         int(support[i]  - tp[i]),
        }

    # ── Aggregate metrics (only over classes present in GT) ───────────────────
    oa          = float(tp[present].sum() / n)
    aa          = float(np.mean([per_class[i]["accuracy"]  for i in present]))
    macro_f1    = float(np.mean([per_class[i]["f1_score"]  for i in present]))
    macro_prec  = float(np.mean([per_class[i]["precision"] for i in present]))
    macro_rec   = float(np.mean([per_class[i]["recall"]    for i in present]))
    macro_iou   = float(np.mean([per_class[i]["iou"]       for i in present]))
    weighted_f1 = float(sum(per_class[i]["f1_score"] * support[i] for i in present) / n)
    freq_iou    = float(sum(per_class[i]["iou"]       * support[i] for i in present) / n)
    kappa       = float(cohen_kappa_score(y_true, y_pred))

    sk_report = classification_report(
        y_true, y_pred,
        labels=list(present),
        target_names=[ALBERTA_CLASSES[i] for i in present],
        zero_division=0,
    )

    return {
        "overall_accuracy":   oa,
        "average_accuracy":   aa,
        "macro_f1":           macro_f1,
        "macro_precision":    macro_prec,
        "macro_recall":       macro_rec,
        "macro_iou":          macro_iou,
        "weighted_f1":        weighted_f1,
        "freq_weighted_iou":  freq_iou,
        "kappa":              kappa,
        "total_samples":      n,
        "gt_valid_pixels":    n_gt_valid,
        "pred_unmapped":      n_pred_unmapped,
        "present_classes":    [int(i) for i in present],
        "per_class":          per_class,
        "confusion_matrix":   cm.tolist(),
        "sklearn_report":     sk_report,
    }


# ==================== OUTPUT ====================

def print_metrics(metrics: dict, title: str = "Results"):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  Overall Accuracy  (OA) : {metrics['overall_accuracy']:.6f}  ({metrics['overall_accuracy']*100:.3f}%)")
    print(f"  Average Accuracy  (AA) : {metrics['average_accuracy']:.6f}  ({metrics['average_accuracy']*100:.3f}%)")
    print(f"  Macro F1               : {metrics['macro_f1']:.6f}")
    print(f"  Weighted F1            : {metrics['weighted_f1']:.6f}")
    print(f"  Macro IoU              : {metrics['macro_iou']:.6f}")
    print(f"  Freq-Weighted IoU      : {metrics['freq_weighted_iou']:.6f}")
    print(f"  Cohen's Kappa          : {metrics['kappa']:.6f}")
    print(f"  Macro Precision        : {metrics['macro_precision']:.6f}")
    print(f"  Macro Recall           : {metrics['macro_recall']:.6f}")
    print(f"  Total valid samples    : {metrics['total_samples']:,}")
    print(f"  GT valid pixels        : {metrics['gt_valid_pixels']:,}")
    print(f"  Pred unmapped pixels   : {metrics['pred_unmapped']:,}  "
          f"({metrics['pred_unmapped']/metrics['gt_valid_pixels']*100:.4f}%)")

    print(f"\n{'─'*105}")
    print(f"{'Cls':<5} {'Class Name':<35} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>8} {'IoU':>8} {'Support':>12}")
    print(f"{'─'*105}")
    for i in range(NUM_CLASSES):
        c = metrics["per_class"][i]
        if c["support"] == 0:
            print(f"{i:<5} {c['class_name'][:35]:<35} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'0':>12}")
            continue
        print(f"{i:<5} {c['class_name'][:35]:<35} "
              f"{c['accuracy']:>10.4f} {c['precision']:>10.4f} {c['recall']:>10.4f} "
              f"{c['f1_score']:>8.4f} {c['iou']:>8.4f} {c['support']:>12,}")
    print(f"{'─'*105}")
    print(f"\n  [sklearn cross-check]\n{metrics['sklearn_report']}")


def save_txt(metrics: dict, path: Path, model_name: str, sensor_name: str, scene_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".txt")

    with open(path, "w") as f:
        def w(line=""):
            f.write(line + "\n")

        w("=" * 80)
        w("CLASSIFICATION METRICS REPORT — FULL SCENE (TIF-based)")
        w("=" * 80)
        w()
        w(f"Model        : {model_name}")
        w(f"Sensor       : {sensor_name}")
        w(f"Scene file   : {scene_name}")
        w(f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"Label space  : Canada-wide IDs remapped to Alberta (0-12)")
        w()
        w("CONDITIONS APPLIED")
        w("-" * 50)
        w("  1. Spatial size of GT and TIF must match")
        w("  2. Valid mask derived from GT (-99 = background); applied to both arrays")
        w("  3. Values remapped to [0-12]: GT verified, TIF remapped via CANADA_TO_ALBERTA")
        w()
        w("OVERALL METRICS")
        w("-" * 50)
        w(f"  Overall Accuracy  (OA) : {metrics['overall_accuracy']:.6f}  ({metrics['overall_accuracy']*100:.3f}%)")
        w(f"  Average Accuracy  (AA) : {metrics['average_accuracy']:.6f}  ({metrics['average_accuracy']*100:.3f}%)")
        w(f"  Macro F1               : {metrics['macro_f1']:.6f}")
        w(f"  Weighted F1            : {metrics['weighted_f1']:.6f}")
        w(f"  Macro IoU              : {metrics['macro_iou']:.6f}")
        w(f"  Freq-Weighted IoU      : {metrics['freq_weighted_iou']:.6f}")
        w(f"  Cohen's Kappa          : {metrics['kappa']:.6f}")
        w(f"  Macro Precision        : {metrics['macro_precision']:.6f}")
        w(f"  Macro Recall           : {metrics['macro_recall']:.6f}")
        w(f"  Total valid samples    : {metrics['total_samples']:,}")
        w(f"  GT valid pixels        : {metrics['gt_valid_pixels']:,}")
        w(f"  Pred unmapped pixels   : {metrics['pred_unmapped']:,}  "
          f"({metrics['pred_unmapped']/metrics['gt_valid_pixels']*100:.4f}%)")
        w()
        w("PER-CLASS METRICS")
        w("-" * 105)
        w(f"{'Cls':<5} {'Class Name':<35} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>8} {'IoU':>8} {'Support':>12}")
        w("-" * 105)
        for i in range(NUM_CLASSES):
            c = metrics["per_class"][i]
            if c["support"] == 0:
                w(f"{i:<5} {c['class_name'][:35]:<35} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'0':>12}")
                continue
            w(f"{i:<5} {c['class_name'][:35]:<35} "
              f"{c['accuracy']:>10.4f} {c['precision']:>10.4f} {c['recall']:>10.4f} "
              f"{c['f1_score']:>8.4f} {c['iou']:>8.4f} {c['support']:>12,}")
        w("-" * 105)
        w()
        w("SKLEARN CLASSIFICATION REPORT")
        w("-" * 50)
        w(metrics["sklearn_report"])
        w()
        w("CONFUSION MATRIX  (rows=true label, cols=predicted label)")
        w("-" * 60)
        w("     " + "".join(f"{j:8d}" for j in range(NUM_CLASSES)))
        for i in range(NUM_CLASSES):
            w(f"{i:4d} " + "".join(f"{metrics['confusion_matrix'][i][j]:8d}"
                                    for j in range(NUM_CLASSES)))
        w()
        w("=" * 80)
        w("END OF REPORT")
        w("=" * 80)

    print(f"✓ TXT saved → {path}")
    return path


def save_json(metrics: dict, path: Path, model_name: str, sensor_name: str, scene_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".json")
    out = {
        "model_name":  model_name,
        "sensor_name": sensor_name,
        "scene_file":  scene_name,
        "timestamp":   datetime.now().isoformat(),
        **{k: v for k, v in metrics.items() if k != "sklearn_report"},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON saved → {path}")


# ==================== EXPERIMENT DISCOVERY ====================

def find_latest_experiment_dir(experiments_base: str, model_name: str, sensor_name: str):
    base    = Path(experiments_base)
    matches = []
    for d in base.iterdir():
        if not d.is_dir() or not d.name.endswith("_v2"):
            continue
        name_l = d.name.lower()
        if model_name.lower() in name_l and sensor_name.lower() in name_l:
            matches.append(d)

    if not matches:
        print(f"❌ No experiment dirs for model={model_name}, sensor={sensor_name}")
        return None

    def ts(name):
        parts = name.split("_")
        for i, p in enumerate(parts):
            if (len(p) == 8 and p.isdigit()
                    and i + 1 < len(parts)
                    and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()):
                return int(p + parts[i + 1])
        return 0

    matches.sort(key=lambda d: ts(d.name), reverse=True)
    print(f"\n📁 Found {len(matches)} experiment dir(s):")
    for i, d in enumerate(matches):
        print(f"   {i+1}. {d.name}")
    print(f"✅ Using: {matches[0].name}")
    return matches[0]


def find_labels_tif(experiment_dir: Path):
    classified_dir = experiment_dir / "classified_scenes"
    if not classified_dir.exists():
        return []
    return list(classified_dir.glob("*_CLIPPED.tif"))


# ==================== MAIN ====================

def main():
    filter_window_size= 3
    parser = argparse.ArgumentParser(
        description="Full-scene classification metrics from a classified labels TIF"
    )
    parser.add_argument(
        "--experiments_base",
        default="/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/experiments",
    )
    parser.add_argument(
        "--ground_truth",
        # default="/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/"
        #         "GroundTruth_Landsat_Canada/landcover-2020-classification_CLIPPED_ALBERTA_REMAPPED.tif",
        default=f"/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/filtered_ground_truth/LC_filtered_{filter_window_size}x{filter_window_size}.tif"
    )
    parser.add_argument("--model_name",  required=True)
    parser.add_argument("--sensor_name", required=True,
                        choices=["landsat8", "sentinel2", "alphaearth"])
    parser.add_argument(
        "--tif_path", default=None,
        help="Optional: explicit path to the *_CLIPPED.tif. "
             "Auto-discovered from the experiment dir if omitted.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Model  : {args.model_name}")
    print(f" Sensor : {args.sensor_name}")
    print(f"{'='*60}")

    if not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        return

    # ── Resolve TIF path(s) ───────────────────────────────────────────────────
    if args.tif_path:
        tif_files = [Path(args.tif_path)]
        exp_dir   = Path(args.tif_path).parent.parent   # for summary CSV location
    else:
        exp_dir   = find_latest_experiment_dir(
            args.experiments_base, args.model_name, args.sensor_name
        )
        if exp_dir is None:
            return
        tif_files = find_labels_tif(exp_dir)
        if not tif_files:
            print("❌ No *_CLIPPED.tif found in classified_scenes/")
            return

    print(f"\n📊 {len(tif_files)} TIF file(s) to process:")
    for f in tif_files:
        print(f"   {f.name}")

    # ── Load ground truth once ────────────────────────────────────────────────
    gt, gt_transform, gt_crs, gt_bounds = load_ground_truth(args.ground_truth)

    all_rows = []

    for tif_path in tif_files:
        print(f"\n{'='*80}")
        print(f" Processing: {tif_path.name}")
        print(f"{'='*80}")

        try:
            # ── Load + remap TIF (Condition 3, TIF side) ──────────────────────
            pred, pred_transform, pred_crs, pred_bounds = load_and_remap_tif(str(tif_path))

            # ── Alignment check ───────────────────────────────────────────────
            check_alignment(
                gt_transform, gt_crs, gt_bounds,
                pred_transform, pred_crs, pred_bounds,
                label=tif_path.name,
            )

            # ── Compute metrics (Conditions 1, 2, 3 all verified inside) ─────
            metrics = compute_metrics(gt, pred)
            print_metrics(metrics, title=tif_path.stem)

            # ── Save outputs ──────────────────────────────────────────────────
            out_stem = tif_path.stem + "_metrics"
            out_dir  = tif_path.parent.parent / f"full_{filter_window_size}x{filter_window_size}filtered_scene_metrics"
            save_txt( metrics, out_dir / out_stem,
                      args.model_name, args.sensor_name, tif_path.name)
            save_json(metrics, out_dir / out_stem,
                      args.model_name, args.sensor_name, tif_path.name)

            all_rows.append({
                "model_name":       args.model_name,
                "sensor_name":      args.sensor_name,
                "scene_file":       tif_path.name,
                "overall_accuracy": metrics["overall_accuracy"],
                "average_accuracy": metrics["average_accuracy"],
                "macro_f1":         metrics["macro_f1"],
                "weighted_f1":      metrics["weighted_f1"],
                "macro_iou":        metrics["macro_iou"],
                "kappa":            metrics["kappa"],
                "total_samples":    metrics["total_samples"],
                "gt_valid_pixels":  metrics["gt_valid_pixels"],
                "pred_unmapped":    metrics["pred_unmapped"],
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows).sort_values("overall_accuracy", ascending=False)
        print(f"\n{'='*80}")
        print(" SUMMARY")
        print(f"{'='*80}")
        print(df.to_string(index=False))

        summary_path = (
            exp_dir / "classified_scenes"
            / f"{args.model_name}_{args.sensor_name}_tif_metrics_summary.csv"
        )
        df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary CSV → {summary_path}")


if __name__ == "__main__":
    main()
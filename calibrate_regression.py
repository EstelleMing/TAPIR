#!/usr/bin/env python3
"""回归仿射校准：仅在 valid 拟合 a,b，再评 test；不改分类结构。

\hat y_cal = clip(a * y_hat + b, -3, 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import MOSIAlignedDataset, collate_eval, load_aligned_split  # noqa: E402
from models.tapir import TAPIR  # noqa: E402
from train_extended import (  # noqa: E402
    CKPT_DIR,
    CLS_NAMES,
    LOG_DIR,
    OUT_DIR,
    compute_metrics_full,
    make_loader,
    predict_split,
)

FIG_DIR = OUT_DIR / "calibration"


def weighted_f1(y_cls: np.ndarray, pred_cls: np.ndarray) -> float:
    """按各类 support 加权的 F1（Weighted-F1）。"""
    y_cls = y_cls.astype(np.int64)
    pred_cls = pred_cls.astype(np.int64)
    total = len(y_cls)
    wsum = 0.0
    for c in range(3):
        tp = np.sum((pred_cls == c) & (y_cls == c))
        fp = np.sum((pred_cls == c) & (y_cls != c))
        fn = np.sum((pred_cls != c) & (y_cls == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        support = int(np.sum(y_cls == c))
        wsum += f1 * support
    return float(wsum / (total + 1e-12))


def enrich_metrics(
    y_cls: np.ndarray, pred_cls: np.ndarray, y_reg: np.ndarray, pred_reg: np.ndarray
) -> Dict[str, Any]:
    m = compute_metrics_full(y_cls, pred_cls, y_reg, pred_reg)
    m["Weighted-F1"] = weighted_f1(y_cls, pred_cls)
    return m


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """最小二乘：y ≈ a * pred + b。"""
    a, b = np.polyfit(y_pred.astype(np.float64), y_true.astype(np.float64), deg=1)
    return float(a), float(b)


def apply_affine(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a * y_pred + b, -3.0, 3.0).astype(np.float32)


def group_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """按真实回归值分 Neg / Zero / Pos 三组看误差。"""
    groups = {
        "neg_lt_-0.5": y_true < -0.5,
        "near_zero_abs_le_0.5": np.abs(y_true) <= 0.5,
        "pos_gt_0.5": y_true > 0.5,
    }
    out = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        err = yp - yt
        out[name] = {
            "n": int(mask.sum()),
            "y_mean": float(yt.mean()),
            "pred_mean": float(yp.mean()),
            "bias": float(err.mean()),
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
        }
    return out


def dist_stats(name: str, y: np.ndarray) -> Dict[str, float]:
    return {
        "name": name,
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "p10": float(np.percentile(y, 10)),
        "p50": float(np.percentile(y, 50)),
        "p90": float(np.percentile(y, 90)),
    }


def plot_scatter(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
    split: str,
    a: float,
    b: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
    for ax, yp, title in [
        (axes[0], y_raw, f"{split} raw"),
        (axes[1], y_cal, f"{split} calibrated"),
    ]:
        ax.scatter(y_true, yp, s=12, alpha=0.45, edgecolors="none")
        ax.plot([-3, 3], [-3, 3], "k--", lw=1, label="y=x")
        mae = float(np.mean(np.abs(yp - y_true)))
        # Corr
        p = yp - yp.mean()
        t = y_true - y_true.mean()
        corr = float((p * t).sum() / (np.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-12))
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("true regression")
        ax.set_ylabel("predicted")
        ax.set_title(f"{title}\nMAE={mae:.4f} Corr={corr:.4f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"affine: y_cal=clip({a:.4f}*y_hat+{b:.4f}, -3, 3)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="valid 仿射回归校准 → test 评测")
    p.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "ours_selected_best.pt"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bert_path", type=str, default=None)
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_aligned_split()
    valid_ds = MOSIAlignedDataset(data["valid"], "valid", train_mode=False)
    test_ds = MOSIAlignedDataset(data["test"], "test", train_mode=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = TAPIR(d=args.hidden, bert_path=args.bert_path).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    yc_v, pc_v, yr_v, pr_v = predict_split(model, make_loader(valid_ds, args.batch_size), device)
    yc_t, pc_t, yr_t, pr_t = predict_split(model, make_loader(test_ds, args.batch_size), device)

    a, b = fit_affine(yr_v, pr_v)
    pr_v_cal = apply_affine(pr_v, a, b)
    pr_t_cal = apply_affine(pr_t, a, b)

    raw_valid = enrich_metrics(yc_v, pc_v, yr_v, pr_v)
    cal_valid = enrich_metrics(yc_v, pc_v, yr_v, pr_v_cal)
    raw_test = enrich_metrics(yc_t, pc_t, yr_t, pr_t)
    cal_test = enrich_metrics(yc_t, pc_t, yr_t, pr_t_cal)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_scatter(yr_v, pr_v, pr_v_cal, "valid", a, b, FIG_DIR / "valid_scatter.png")
    plot_scatter(yr_t, pr_t, pr_t_cal, "test", a, b, FIG_DIR / "test_scatter.png")

    report = {
        "ckpt": str(args.ckpt),
        "affine": {"a": a, "b": b, "formula": "clip(a*y_hat+b, -3, 3)", "fit_on": "valid"},
        "distribution": {
            "valid_true": dist_stats("valid_true", yr_v),
            "valid_pred_raw": dist_stats("valid_pred_raw", pr_v),
            "valid_pred_cal": dist_stats("valid_pred_cal", pr_v_cal),
            "test_true": dist_stats("test_true", yr_t),
            "test_pred_raw": dist_stats("test_pred_raw", pr_t),
            "test_pred_cal": dist_stats("test_pred_cal", pr_t_cal),
        },
        "group_errors": {
            "valid_raw": group_errors(yr_v, pr_v),
            "valid_cal": group_errors(yr_v, pr_v_cal),
            "test_raw": group_errors(yr_t, pr_t),
            "test_cal": group_errors(yr_t, pr_t_cal),
        },
        "metrics": {
            "valid_raw": {k: raw_valid[k] for k in ("Acc", "Macro-F1", "Weighted-F1", "MAE", "Corr")},
            "valid_cal": {k: cal_valid[k] for k in ("Acc", "Macro-F1", "Weighted-F1", "MAE", "Corr")},
            "test_raw": {
                k: raw_test[k] for k in ("Acc", "Macro-F1", "Weighted-F1", "MAE", "Corr")
            },
            "test_cal": {
                k: cal_test[k] for k in ("Acc", "Macro-F1", "Weighted-F1", "MAE", "Corr")
            },
        },
        "test_per_class_f1": {
            name: raw_test["per_class"][name]["F1"] for name in CLS_NAMES
        },
        "figures": {
            "valid": str(FIG_DIR / "valid_scatter.png"),
            "test": str(FIG_DIR / "test_scatter.png"),
        },
        "note": "校准仅改回归值；分类 logits/Acc/F1 不变。",
    }

    out_json = LOG_DIR / "regression_calibration.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("==== affine (fit on valid) ====")
    print(f"a={a:.6f}  b={b:.6f}")
    print("==== distribution (mean / std) ====")
    for key in ("valid_true", "valid_pred_raw", "valid_pred_cal", "test_true", "test_pred_raw", "test_pred_cal"):
        d = report["distribution"][key]
        print(f"  {key:18s} mean={d['mean']:+.4f} std={d['std']:.4f}")
    print("==== metrics ====")
    for split in ("valid_raw", "valid_cal", "test_raw", "test_cal"):
        m = report["metrics"][split]
        print(
            f"  {split:12s} Acc={m['Acc']:.4f} MacroF1={m['Macro-F1']:.4f} "
            f"W-F1={m['Weighted-F1']:.4f} MAE={m['MAE']:.4f} Corr={m['Corr']:.4f}"
        )
    print("==== test group errors (raw → cal) ====")
    for g in report["group_errors"]["test_raw"]:
        r = report["group_errors"]["test_raw"][g]
        c = report["group_errors"]["test_cal"][g]
        print(
            f"  {g:24s} n={r['n']:3d}  "
            f"bias {r['bias']:+.3f}→{c['bias']:+.3f}  "
            f"MAE {r['MAE']:.3f}→{c['MAE']:.3f}"
        )
    print(f"saved {out_json}")
    print(f"figs  {FIG_DIR}")


if __name__ == "__main__":
    main()

"""延长训练诊断：对比 BERT 冻结轮数，按 valid 综合分早停；test 仅最终评一次。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import (  # noqa: E402
    MOSIAlignedDataset,
    class_weights_from_labels,
    collate_eval,
    collate_train,
    load_aligned_split,
)
from core.loss import Problem2Criterion  # noqa: E402
from core.masks import load_eval_masks  # noqa: E402
from core.utils import overlay_yaml  # noqa: E402
from models.tapir import TAPIR  # noqa: E402

LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"
MASK_DIR = ROOT / "masks"
OUT_DIR = ROOT / "outputs"
CLS_NAMES = ["Negative", "Neutral", "Positive"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def composite_score(acc: float, macro_f1: float) -> float:
    """验证集选模指标：0.5 * Acc + 0.5 * Macro-F1。"""
    return 0.5 * acc + 0.5 * macro_f1


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 3) -> List[List[int]]:
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= t < n_class and 0 <= p < n_class:
            cm[t, p] += 1
    return cm.tolist()


def compute_metrics_full(
    y_cls: np.ndarray, pred_cls: np.ndarray, y_reg: np.ndarray, pred_reg: np.ndarray
) -> Dict[str, Any]:
    """Acc / Macro-F1 / 逐类 P/R/F1 / MAE / Corr / 混淆矩阵。"""
    y_cls = y_cls.astype(np.int64)
    pred_cls = pred_cls.astype(np.int64)
    acc = float((y_cls == pred_cls).mean())
    per_class = {}
    f1s = []
    for c, name in enumerate(CLS_NAMES):
        tp = int(np.sum((pred_cls == c) & (y_cls == c)))
        fp = int(np.sum((pred_cls == c) & (y_cls != c)))
        fn = int(np.sum((pred_cls != c) & (y_cls == c)))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
        per_class[name] = {
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "support": int(np.sum(y_cls == c)),
        }
    macro_f1 = float(np.mean(f1s))
    mae = float(np.mean(np.abs(pred_reg - y_reg)))
    p = pred_reg - pred_reg.mean()
    t = y_reg - y_reg.mean()
    corr = float((p * t).sum() / (np.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-12))
    return {
        "Acc": acc,
        "Macro-F1": macro_f1,
        "MAE": mae,
        "Corr": corr,
        "composite": composite_score(acc, macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_cls, pred_cls),
        "confusion_matrix_note": "rows=true[Neg,Neu,Pos], cols=pred[Neg,Neu,Pos]",
    }


@torch.no_grad()
def predict_split(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_yc, all_pc, all_yr, all_pr = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        all_yc.append(batch["y_cls"].cpu().numpy())
        all_pc.append(out["logits"].argmax(-1).cpu().numpy())
        all_yr.append(batch["y_reg"].cpu().numpy())
        all_pr.append(out["y_hat"].cpu().numpy())
    return (
        np.concatenate(all_yc),
        np.concatenate(all_pc),
        np.concatenate(all_yr),
        np.concatenate(all_pr),
    )


def evaluate_full(model, loader, device) -> Dict[str, Any]:
    return compute_metrics_full(*predict_split(model, loader, device))


def build_optimizer(model, lr_new: float, lr_bert: float, wd: float):
    bert_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.bert" in n:
            bert_params.append(p)
        else:
            other_params.append(p)
    groups = [{"params": other_params, "lr": lr_new}]
    if bert_params:
        groups.append({"params": bert_params, "lr": lr_bert})
    return torch.optim.AdamW(groups, weight_decay=wd)


def update_ema(teacher, student, momentum: float = 0.995):
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def ensure_valid_masks(text_berts: np.ndarray, seed: int = 2026) -> Dict[float, Dict[str, np.ndarray]]:
    """为 valid 预生成固定缺失掩码（独立文件名，不覆盖 test 掩码）。"""
    from core.masks import effective_length, sample_contiguous_mask

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    packs = {}
    for r in (0.3, 0.5):
        path = MASK_DIR / f"valid_mask_r{r:.1f}_seed{seed}.npz"
        if not path.exists():
            N, _, T = text_berts.shape
            # offset 与 test 区分，保证可复现且互不覆盖
            rng = np.random.default_rng(seed + int(r * 1000) + 777)
            Mt = np.zeros((N, T), dtype=np.float32)
            Ma = np.zeros((N, T), dtype=np.float32)
            Mv = np.zeros((N, T), dtype=np.float32)
            for i in range(N):
                L = effective_length(text_berts[i, 1])
                mt, ma, mv = sample_contiguous_mask(
                    L, T, float(r), rng, sync_modalities=("t", "a", "v")
                )
                Mt[i], Ma[i], Mv[i] = mt, ma, mv
            np.savez_compressed(path, Mt=Mt, Ma=Ma, Mv=Mv, rate=np.array([r], dtype=np.float32))
            print(f"[masks] saved {path} mean_Mt={Mt.mean():.4f}")
        packs[r] = load_eval_masks(path)
    return packs


def make_loader(ds, batch_size: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_eval)


def flatten_epoch_row(hist: Dict[str, Any]) -> Dict[str, Any]:
    """把一轮 history 展平为 csv 行。"""
    v = hist["valid_clean"]
    row = {
        "epoch": hist["epoch"],
        "train_loss": hist["train_loss"],
        "sec": hist["sec"],
        "Acc": v["Acc"],
        "Macro-F1": v["Macro-F1"],
        "composite": v["composite"],
        "MAE": v["MAE"],
        "Corr": v["Corr"],
        "P_neg": v["per_class"]["Negative"]["Precision"],
        "R_neg": v["per_class"]["Negative"]["Recall"],
        "F1_neg": v["per_class"]["Negative"]["F1"],
        "P_neu": v["per_class"]["Neutral"]["Precision"],
        "R_neu": v["per_class"]["Neutral"]["Recall"],
        "F1_neu": v["per_class"]["Neutral"]["F1"],
        "P_pos": v["per_class"]["Positive"]["Precision"],
        "R_pos": v["per_class"]["Positive"]["Recall"],
        "F1_pos": v["per_class"]["Positive"]["F1"],
        "miss03_Macro-F1": hist["valid_miss_0.3"]["Macro-F1"],
        "miss03_MAE": hist["valid_miss_0.3"]["MAE"],
        "miss05_Macro-F1": hist["valid_miss_0.5"]["Macro-F1"],
        "miss05_MAE": hist["valid_miss_0.5"]["MAE"],
        "bert_mode": hist["bert_mode"],
    }
    return row


def train_one_schedule(
    args,
    data,
    freeze_epochs: int,
    valid_masks: Dict[float, Dict[str, np.ndarray]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """从零训练一个冻结日程；不在此处评 test。"""
    tag = f"freeze{freeze_epochs}"
    device = torch.device(args.device)
    set_seed(args.seed)  # 每个日程独立可复现

    train_ds = MOSIAlignedDataset(data["train"], "train", train_mode=True, seed=args.seed)
    valid_ds = MOSIAlignedDataset(data["valid"], "valid", train_mode=False)
    valid_m03 = MOSIAlignedDataset(
        data["valid"], "valid", train_mode=False, external_masks=valid_masks[0.3]
    )
    valid_m05 = MOSIAlignedDataset(
        data["valid"], "valid", train_mode=False, external_masks=valid_masks[0.5]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_train,
        pin_memory=True,
    )
    valid_loader = make_loader(valid_ds, args.batch_size)
    loader_03 = make_loader(valid_m03, args.batch_size)
    loader_05 = make_loader(valid_m05, args.batch_size)

    model = TAPIR(d=args.hidden, bert_path=args.bert_path).to(device)
    teacher = copy.deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    cw = class_weights_from_labels(data["train"]["classification_labels"]).to(device)
    criterion = Problem2Criterion(cw, w_elastic=0.0).to(device)

    best_score = -1.0
    best_state = None
    best_epoch = -1
    best_valid = None
    best_miss = None
    patience_left = args.patience
    history: List[Dict[str, Any]] = []
    early_stopped = False
    epochs_ran = 0
    oom = False
    prev_bert_mode = None

    logger.info(f"===== start schedule {tag}: freeze_epochs={freeze_epochs}, "
                f"max_epochs={args.epochs}, patience={args.patience} =====")

    for epoch in range(1, args.epochs + 1):
        bert_mode = "freeze" if epoch <= freeze_epochs else "last2"
        model.set_bert_trainable(bert_mode)
        if bert_mode != prev_bert_mode:
            optim = build_optimizer(model, args.lr_new, args.lr_bert, args.wd)
            prev_bert_mode = bert_mode
            logger.info(f"[{tag}] epoch {epoch}: BERT mode -> {bert_mode}, rebuild optimizer")
        elif epoch == 1:
            optim = build_optimizer(model, args.lr_new, args.lr_bert, args.wd)

        model.train()
        t0 = time.time()
        loss_meter = []
        try:
            for step, views in enumerate(train_loader):
                full = {k: v.to(device) for k, v in views["full"].items()}
                m1 = {k: v.to(device) for k, v in views["M1"].items()}
                m2 = {k: v.to(device) for k, v in views["M2"].items()}
                out_f = model(full)
                out_1 = model(m1)
                out_2 = model(m2)
                with torch.no_grad():
                    t_out = teacher(full)
                loss, stats = criterion(out_f, out_1, out_2, full, m1, m2, teacher_out=t_out)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                update_ema(teacher, model, args.ema)
                loss_meter.append(stats["loss"])
                if step % 20 == 0:
                    logger.info(
                        f"[{tag}] ep{epoch} step{step}/{len(train_loader)} "
                        f"loss={stats['loss']:.4f}"
                    )
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            logger.error(f"[{tag}] OOM at epoch {epoch}: {e}")
            torch.cuda.empty_cache()
            break

        epochs_ran = epoch
        val_clean = evaluate_full(model, valid_loader, device)
        val_03 = evaluate_full(model, loader_03, device)
        val_05 = evaluate_full(model, loader_05, device)
        tr_loss = float(np.mean(loss_meter)) if loss_meter else float("nan")
        hist = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "sec": time.time() - t0,
            "bert_mode": bert_mode,
            "valid_clean": val_clean,
            "valid_miss_0.3": {
                "Macro-F1": val_03["Macro-F1"],
                "MAE": val_03["MAE"],
                "Acc": val_03["Acc"],
                "Corr": val_03["Corr"],
            },
            "valid_miss_0.5": {
                "Macro-F1": val_05["Macro-F1"],
                "MAE": val_05["MAE"],
                "Acc": val_05["Acc"],
                "Corr": val_05["Corr"],
            },
        }
        history.append(hist)
        score = val_clean["composite"]
        logger.info(
            f"[{tag}] epoch {epoch} done loss={tr_loss:.4f} "
            f"Acc={val_clean['Acc']:.4f} F1={val_clean['Macro-F1']:.4f} "
            f"comp={score:.4f} MAE={val_clean['MAE']:.4f} "
            f"miss0.3 F1={val_03['Macro-F1']:.4f} miss0.5 F1={val_05['Macro-F1']:.4f} "
            f"neuF1={val_clean['per_class']['Neutral']['F1']:.4f} "
            f"time={hist['sec']:.1f}s"
        )

        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_valid = val_clean
            best_miss = {"0.3": hist["valid_miss_0.3"], "0.5": hist["valid_miss_0.5"]}
            patience_left = args.patience
            ckpt_path = CKPT_DIR / f"{run_prefix(args)}_{tag}_best.pt"
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "epoch": epoch,
                    "valid": val_clean,
                    "valid_miss": best_miss,
                    "freeze_epochs": freeze_epochs,
                    "composite": score,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"[{tag}] new best composite={score:.4f} -> {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_stopped = True
                logger.info(f"[{tag}] early stop at epoch {epoch} (best={best_epoch})")
                break

    # 写逐轮 json/csv
    json_path = LOG_DIR / f"{run_prefix(args)}_{tag}_history.json"
    csv_path = LOG_DIR / f"{run_prefix(args)}_{tag}_history.csv"
    result = {
        "tag": tag,
        "freeze_epochs": freeze_epochs,
        "best_epoch": best_epoch,
        "best_composite": best_score,
        "best_valid_clean": best_valid,
        "best_valid_miss": best_miss,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "oom": oom,
        "history": history,
        "note": "test 未在此日程内评测；由 valid 综合分选模后统一评一次",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        rows = [flatten_epoch_row(h) for h in history]
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    logger.info(f"[{tag}] history -> {json_path} / {csv_path}")

    # 释放显存
    del model, teacher
    torch.cuda.empty_cache()
    return result


def run_prefix(args) -> str:
    name = str(getattr(args, "run_name", "") or "ours").strip()
    return name or "ours"


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """二分类 Acc 与按 support 加权的 F1。标签取 0/1。"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    f1s, supports = [], []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        supports.append(int(np.sum(y_true == c)))
    w = float(np.dot(f1s, supports) / (sum(supports) + 1e-12))
    return {"Acc": acc, "F1": w, "n": int(len(y_true)), "support": supports}


def polarity_bundle(y_cls, pred_cls, y_reg, pred_reg) -> Dict[str, Any]:
    """三分类之外的两种二分类口径。"""
    # 排除真值中性；预测为中性时用回归符号
    keep = y_cls != 1
    yt = (y_cls[keep] == 2).astype(np.int64)
    pc = pred_cls[keep]
    pr = pred_reg[keep]
    yp = np.where(pc == 2, 1, np.where(pc == 0, 0, (pr > 0).astype(np.int64)))
    # 负 (<0) vs 非负 (>=0)，用回归值
    yt_has0 = (y_reg >= 0).astype(np.int64)
    yp_has0 = (pred_reg >= 0).astype(np.int64)
    return {
        "exclude_neutral": binary_metrics(yt, yp),
        "nonneg_vs_neg": binary_metrics(yt_has0, yp_has0),
    }


def weighted_f1(y_cls: np.ndarray, pred_cls: np.ndarray) -> float:
    y_cls = y_cls.astype(np.int64)
    pred_cls = pred_cls.astype(np.int64)
    total = 0.0
    for c in range(3):
        tp = np.sum((pred_cls == c) & (y_cls == c))
        fp = np.sum((pred_cls == c) & (y_cls != c))
        fn = np.sum((pred_cls != c) & (y_cls == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        total += f1 * np.sum(y_cls == c)
    return float(total / (len(y_cls) + 1e-12))


def final_test_eval(
    args,
    data,
    selected: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """仅对 valid 选定的配置评一次 test。"""
    tag = selected["tag"]
    device = torch.device(args.device)
    ckpt_path = CKPT_DIR / f"{run_prefix(args)}_{tag}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = TAPIR(d=args.hidden, bert_path=args.bert_path).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    valid_ds = MOSIAlignedDataset(data["valid"], "valid", train_mode=False)
    test_ds = MOSIAlignedDataset(data["test"], "test", train_mode=False)
    yc_v, pc_v, yr_v, pr_v = predict_split(model, make_loader(valid_ds, args.batch_size), device)
    yc_t, pc_t, yr_t, pr_t = predict_split(model, make_loader(test_ds, args.batch_size), device)
    valid_m = compute_metrics_full(yc_v, pc_v, yr_v, pr_v)
    test_m = compute_metrics_full(yc_t, pc_t, yr_t, pr_t)
    valid_m["Weighted-F1"] = weighted_f1(yc_v, pc_v)
    test_m["Weighted-F1"] = weighted_f1(yc_t, pc_t)

    # 仿射校准只在 valid 上拟合
    a_coef, b_coef = np.polyfit(pr_v.astype(np.float64), yr_v.astype(np.float64), deg=1)
    pr_v_cal = np.clip(a_coef * pr_v + b_coef, -3.0, 3.0).astype(np.float32)
    pr_t_cal = np.clip(a_coef * pr_t + b_coef, -3.0, 3.0).astype(np.float32)
    valid_cal = compute_metrics_full(yc_v, pc_v, yr_v, pr_v_cal)
    test_cal = compute_metrics_full(yc_t, pc_t, yr_t, pr_t_cal)

    out = {
        "selected_tag": tag,
        "run_name": run_prefix(args),
        "freeze_epochs": selected["freeze_epochs"],
        "best_epoch": selected["best_epoch"],
        "valid": valid_m,
        "test": test_m,
        "binary": {
            "valid": polarity_bundle(yc_v, pc_v, yr_v, pr_v),
            "test": polarity_bundle(yc_t, pc_t, yr_t, pr_t),
        },
        "regression_calibration": {
            "a": float(a_coef),
            "b": float(b_coef),
            "fit_on": "valid",
            "valid_MAE": valid_cal["MAE"],
            "valid_Corr": valid_cal["Corr"],
            "test_MAE": test_cal["MAE"],
            "test_Corr": test_cal["Corr"],
        },
        "valid_miss_at_best": selected.get("best_valid_miss"),
    }
    path = LOG_DIR / f"{run_prefix(args)}_final.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(
        f"[final] selected={tag} test Acc={test_m['Acc']:.4f} "
        f"Macro-F1={test_m['Macro-F1']:.4f} W-F1={test_m['Weighted-F1']:.4f} "
        f"MAE={test_m['MAE']:.4f}->{test_cal['MAE']:.4f}"
    )
    logger.info(f"[final] saved {path}")

    torch.save(ckpt, CKPT_DIR / f"{run_prefix(args)}_best.pt")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr_new", type=float, default=1e-4)
    p.add_argument("--lr_bert", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bert_path", type=str, default=None)
    p.add_argument(
        "--freeze_list",
        type=int,
        nargs="+",
        default=[2, 5],
        help="依次训练的冻结轮数列表，默认先 2 再 5",
    )
    p.add_argument("--mask_seed", type=int, default=2026)
    p.add_argument(
        "--only_select_and_test",
        action="store_true",
        help="跳过训练，仅根据已有 history 选模并评 test",
    )
    p.add_argument("--run_name", type=str, default="ours", help="日志和权重文件名前缀，避免覆盖旧实验")
    p.add_argument("--config_file", type=str, default="", help="YAML 配置，未在命令行显式给出的字段由此填充")
    args = p.parse_args()
    return overlay_yaml(args, p)


def main():
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ext", LOG_DIR / f"{run_prefix(args)}_train.log")
    logger.info(f"args={vars(args)}")

    data = load_aligned_split()
    valid_masks = ensure_valid_masks(data["valid"]["text_bert"], seed=args.mask_seed)

    schedule_results = []
    if not args.only_select_and_test:
        for fe in args.freeze_list:
            r = train_one_schedule(args, data, fe, valid_masks, logger)
            schedule_results.append(r)
            # 每个日程结束后写中间汇总
            with open(LOG_DIR / f"{run_prefix(args)}_compare.json", "w", encoding="utf-8") as f:
                json.dump(schedule_results, f, ensure_ascii=False, indent=2)
    else:
        for fe in args.freeze_list:
            path = LOG_DIR / f"{run_prefix(args)}_freeze{fe}_history.json"
            with open(path, encoding="utf-8") as f:
                schedule_results.append(json.load(f))

    # 按 valid 最佳综合分选模（不看 test）
    best = max(schedule_results, key=lambda x: x["best_composite"])
    logger.info(
        f"[select] winner={best['tag']} best_epoch={best['best_epoch']} "
        f"comp={best['best_composite']:.4f} "
        f"Acc={best['best_valid_clean']['Acc']:.4f} "
        f"F1={best['best_valid_clean']['Macro-F1']:.4f}"
    )

    final = final_test_eval(args, data, best, logger)

    summary = {
        "schedules": [
            {
                "tag": s["tag"],
                "freeze_epochs": s["freeze_epochs"],
                "best_epoch": s["best_epoch"],
                "best_composite": s["best_composite"],
                "valid_Acc": s["best_valid_clean"]["Acc"],
                "valid_Macro-F1": s["best_valid_clean"]["Macro-F1"],
                "epochs_ran": s["epochs_ran"],
                "early_stopped": s["early_stopped"],
                "valid_miss": s["best_valid_miss"],
            }
            for s in schedule_results
        ],
        "selected": best["tag"],
        "final": final,
    }
    with open(LOG_DIR / f"{run_prefix(args)}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"done -> logs/{run_prefix(args)}_summary.json")


if __name__ == "__main__":
    main()

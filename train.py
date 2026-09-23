"""训练脚本：TAPIR 与 B0 基线。"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from core.loss import BaselineCriterion, Problem2Criterion  # noqa: E402
from core.utils import overlay_yaml  # noqa: E402
from models.tapir import BaselineB0, TAPIR  # noqa: E402

OUT_DIR = ROOT
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"
MASK_DIR = ROOT / "masks"


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


def compute_metrics(
    y_cls: np.ndarray, pred_cls: np.ndarray, y_reg: np.ndarray, pred_reg: np.ndarray
) -> Dict[str, float]:
    y_cls = y_cls.astype(np.int64)
    pred_cls = pred_cls.astype(np.int64)
    acc = float((y_cls == pred_cls).mean())
    f1s = []
    for c in range(3):
        tp = np.sum((pred_cls == c) & (y_cls == c))
        fp = np.sum((pred_cls == c) & (y_cls != c))
        fn = np.sum((pred_cls != c) & (y_cls == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
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
        "F1_neg": float(f1s[0]),
        "F1_neu": float(f1s[1]),
        "F1_pos": float(f1s[2]),
    }


@torch.no_grad()
def evaluate(model, loader, device, is_ours: bool) -> Dict[str, float]:
    model.eval()
    all_yc, all_pc, all_yr, all_pr = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred_c = out["logits"].argmax(dim=-1)
        all_yc.append(batch["y_cls"].cpu().numpy())
        all_pc.append(pred_c.cpu().numpy())
        all_yr.append(batch["y_reg"].cpu().numpy())
        all_pr.append(out["y_hat"].cpu().numpy())
    return compute_metrics(
        np.concatenate(all_yc),
        np.concatenate(all_pc),
        np.concatenate(all_yr),
        np.concatenate(all_pr),
    )


def build_optimizer(model, lr_new: float, lr_bert: float, wd: float):
    bert_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.bert" in n or n.startswith("bert"):
            bert_params.append(p)
        else:
            other_params.append(p)
    groups = [{"params": other_params, "lr": lr_new}]
    if bert_params:
        groups.append({"params": bert_params, "lr": lr_bert})
    return torch.optim.AdamW(groups, weight_decay=wd)


def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float = 0.995):
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def train_ours(args, data, logger) -> Dict:
    device = torch.device(args.device)
    train_ds = MOSIAlignedDataset(data["train"], "train", train_mode=True, seed=args.seed)
    valid_ds = MOSIAlignedDataset(data["valid"], "valid", train_mode=False)
    test_ds = MOSIAlignedDataset(data["test"], "test", train_mode=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_eval
    )

    model = TAPIR(d=args.hidden, bert_path=args.bert_path).to(device)
    teacher = copy.deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    cw = class_weights_from_labels(data["train"]["classification_labels"]).to(device)
    criterion = Problem2Criterion(cw, w_elastic=args.w_elastic).to(device)

    model.set_bert_trainable("freeze")
    optim = build_optimizer(model, args.lr_new, args.lr_bert, args.wd)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_left = args.patience
    history = []
    oom_flag = False
    epochs_ran = 0

    for epoch in range(1, args.epochs + 1):
        # BERT 解冻策略
        if epoch <= 5:
            model.set_bert_trainable("freeze")
        else:
            model.set_bert_trainable("last2")
        # 重建优化器以纳入新参数
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
                        f"[Ours] epoch {epoch} step {step}/{len(train_loader)} "
                        f"loss={stats['loss']:.4f} task={stats['task']:.4f} "
                        f"nll={stats['nll']:.4f} distill={stats['distill']:.4f}"
                    )
        except torch.cuda.OutOfMemoryError as e:
            oom_flag = True
            logger.error(f"OOM at epoch {epoch}: {e}")
            torch.cuda.empty_cache()
            break

        epochs_ran = epoch
        val_m = evaluate(model, valid_loader, device, True)
        tr_loss = float(np.mean(loss_meter)) if loss_meter else float("nan")
        hist = {"epoch": epoch, "train_loss": tr_loss, "valid": val_m, "sec": time.time() - t0}
        history.append(hist)
        logger.info(
            f"[Ours] epoch {epoch} done loss={tr_loss:.4f} "
            f"valid F1={val_m['Macro-F1']:.4f} Acc={val_m['Acc']:.4f} "
            f"MAE={val_m['MAE']:.4f} Corr={val_m['Corr']:.4f} time={hist['sec']:.1f}s"
        )

        if val_m["Macro-F1"] > best_f1:
            best_f1 = val_m["Macro-F1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            ckpt = CKPT_DIR / "ours_best.pt"
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "epoch": epoch, "valid": val_m, "args": vars(args)}, ckpt)
            logger.info(f"[Ours] new best saved -> {ckpt}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"[Ours] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_m = evaluate(model, valid_loader, device, True)
    test_m = evaluate(model, test_loader, device, True)
    result = {
        "model": "Ours",
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "oom": oom_flag,
        "valid_clean": valid_m,
        "test_clean": test_m,
        "history": history,
        "batch_size": args.batch_size,
    }
    out_path = LOG_DIR / "ours_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[Ours] metrics -> {out_path}")
    return result


def train_b0(args, data, logger) -> Dict:
    device = torch.device(args.device)
    # B0 训练：用 full 视图（零填充天然缺失），不做三视图
    train_ds = MOSIAlignedDataset(data["train"], "train", train_mode=False, seed=args.seed)
    valid_ds = MOSIAlignedDataset(data["valid"], "valid", train_mode=False)
    test_ds = MOSIAlignedDataset(data["test"], "test", train_mode=False)

    # 为 B0 增加轻量随机遮挡以增强鲁棒性：复用 train_mode 但只用 M1 作为输入
    # 按方案：零填充 + 同结构；训练时也可采样缺失。这里用 train_mode 的 M1 视图。
    train_ds_aug = MOSIAlignedDataset(data["train"], "train", train_mode=True, seed=args.seed)

    def collate_b0(batch):
        # 50% full / 50% M1
        views = []
        for b in batch:
            if random.random() < 0.5:
                views.append(b["full"])
            else:
                views.append(b["M1"])
        return collate_eval(views)

    train_loader = DataLoader(
        train_ds_aug,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_b0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_eval
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_eval
    )

    model = BaselineB0(d=args.hidden, bert_path=args.bert_path).to(device)
    cw = class_weights_from_labels(data["train"]["classification_labels"]).to(device)
    criterion = BaselineCriterion(cw).to(device)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_left = args.patience
    history = []
    oom_flag = False
    epochs_ran = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            model.set_bert_trainable("freeze")
        else:
            model.set_bert_trainable("last2")
        optim = build_optimizer(model, args.lr_new, args.lr_bert, args.wd)

        model.train()
        t0 = time.time()
        loss_meter = []
        try:
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                loss, stats = criterion(out, batch["y_cls"], batch["y_reg"])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                loss_meter.append(stats["loss"])
                if step % 20 == 0:
                    logger.info(
                        f"[B0] epoch {epoch} step {step}/{len(train_loader)} loss={stats['loss']:.4f}"
                    )
        except torch.cuda.OutOfMemoryError as e:
            oom_flag = True
            logger.error(f"OOM at epoch {epoch}: {e}")
            torch.cuda.empty_cache()
            break

        epochs_ran = epoch
        val_m = evaluate(model, valid_loader, device, False)
        tr_loss = float(np.mean(loss_meter)) if loss_meter else float("nan")
        hist = {"epoch": epoch, "train_loss": tr_loss, "valid": val_m, "sec": time.time() - t0}
        history.append(hist)
        logger.info(
            f"[B0] epoch {epoch} done loss={tr_loss:.4f} "
            f"valid F1={val_m['Macro-F1']:.4f} Acc={val_m['Acc']:.4f} "
            f"MAE={val_m['MAE']:.4f} Corr={val_m['Corr']:.4f} time={hist['sec']:.1f}s"
        )

        if val_m["Macro-F1"] > best_f1:
            best_f1 = val_m["Macro-F1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            ckpt = CKPT_DIR / "b0_best.pt"
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "epoch": epoch, "valid": val_m, "args": vars(args)}, ckpt)
            logger.info(f"[B0] new best saved -> {ckpt}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"[B0] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_m = evaluate(model, valid_loader, device, False)
    test_m = evaluate(model, test_loader, device, False)
    result = {
        "model": "B0",
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "oom": oom_flag,
        "valid_clean": valid_m,
        "test_clean": test_m,
        "history": history,
        "batch_size": args.batch_size,
    }
    out_path = LOG_DIR / "b0_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[B0] metrics -> {out_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ours", "b0", "both"], default="both")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr_new", type=float, default=1e-4)
    p.add_argument("--lr_bert", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--w_elastic", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bert_path", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--try_batch64", action="store_true", help="先尝试 batch=64，OOM 则降到 32/16")
    p.add_argument("--config_file", type=str, default="", help="YAML 配置，未在命令行显式给出的字段由此填充")
    args = p.parse_args()
    return overlay_yaml(args, p)


def main():
    args = parse_args()
    set_seed(args.seed)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("problem2", LOG_DIR / f"train_{args.model}.log")
    logger.info(f"args={vars(args)}")
    logger.info(f"cuda_available={torch.cuda.is_available()}")

    data = load_aligned_split()
    logger.info(
        f"loaded splits train={len(data['train']['id'])} "
        f"valid={len(data['valid']['id'])} test={len(data['test']['id'])}"
    )

    # batch size 自适应
    if args.try_batch64:
        for bs in (64, 32, 16):
            args.batch_size = bs
            logger.info(f"trying batch_size={bs}")
            try:
                # 快速探测
                probe = TAPIR(d=args.hidden, bert_path=args.bert_path).to(args.device)
                dummy = {
                    "input_ids": torch.randint(0, 1000, (bs, 50), device=args.device),
                    "attention_mask": torch.ones(bs, 50, dtype=torch.long, device=args.device),
                    "token_type_ids": torch.zeros(bs, 50, dtype=torch.long, device=args.device),
                    "audio": torch.randn(bs, 50, 74, device=args.device),
                    "vision": torch.randn(bs, 50, 35, device=args.device),
                    "Mt": torch.zeros(bs, 50, device=args.device),
                    "Ma": torch.zeros(bs, 50, device=args.device),
                    "Mv": torch.zeros(bs, 50, device=args.device),
                }
                out = probe(dummy)
                loss = out["logits"].sum() + out["y_hat"].sum()
                loss.backward()
                del probe, out, loss, dummy
                torch.cuda.empty_cache()
                logger.info(f"batch_size={bs} OK")
                break
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"batch_size={bs} OOM, reducing...")
                torch.cuda.empty_cache()
                args.batch_size = 16

    results = {}
    if args.model in ("ours", "both"):
        results["ours"] = train_ours(args, data, logger)
    if args.model in ("b0", "both"):
        results["b0"] = train_b0(args, data, logger)

    summary_path = LOG_DIR / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()

"""在 test 上用预生成掩码评测缺失鲁棒性（Macro-F1 / MAE）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import MOSIAlignedDataset, collate_eval, load_aligned_split  # noqa: E402
from core.masks import generate_eval_masks, load_eval_masks  # noqa: E402
from core.utils import overlay_yaml  # noqa: E402
from models.tapir import BaselineB0, TAPIR  # noqa: E402
from train import compute_metrics  # noqa: E402

MASK_DIR = ROOT / "masks"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    all_yc, all_pc, all_yr, all_pr = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        all_yc.append(batch["y_cls"].cpu().numpy())
        all_pc.append(out["logits"].argmax(-1).cpu().numpy())
        all_yr.append(batch["y_reg"].cpu().numpy())
        all_pr.append(out["y_hat"].cpu().numpy())
    return compute_metrics(
        np.concatenate(all_yc),
        np.concatenate(all_pc),
        np.concatenate(all_yr),
        np.concatenate(all_pr),
    )


def load_model(kind: str, ckpt_path: Path, device: str, bert_path=None):
    if kind == "ours":
        model = TAPIR(bert_path=bert_path)
    else:
        model = BaselineB0(bert_path=bert_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_mask", type=int, default=2026)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--bert_path", type=str, default=None)
    ap.add_argument("--config_file", type=str, default="")
    ap.add_argument("--save_path", type=str, default="", help="结果目录，默认 logs/")
    args = overlay_yaml(ap.parse_args(), ap)

    data = load_aligned_split()
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # 预生成 / 加载掩码
    packs = {}
    for r in args.rates:
        path = MASK_DIR / f"test_mask_r{r:.1f}_seed{args.seed_mask}.npz"
        if not path.exists():
            generate_eval_masks(
                data["test"]["text_bert"],
                rates=[r],
                seed=args.seed_mask,
                out_dir=MASK_DIR,
                sync_modalities=("t", "a", "v"),
            )
        packs[r] = load_eval_masks(path)

    results = {}
    for kind, ckpt_name in [("ours", "ours_best.pt"), ("b0", "b0_best.pt")]:
        ckpt = CKPT_DIR / ckpt_name
        if not ckpt.exists():
            print(f"[skip] missing {ckpt}")
            continue
        model = load_model(kind, ckpt, args.device, args.bert_path)
        results[kind] = {}
        for r, pack in packs.items():
            ds = MOSIAlignedDataset(
                data["test"], "test", train_mode=False, external_masks=pack
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval
            )
            m = eval_loader(model, loader, args.device)
            results[kind][f"rate_{r}"] = {
                "Macro-F1": m["Macro-F1"],
                "MAE": m["MAE"],
                "Acc": m["Acc"],
                "Corr": m["Corr"],
            }
            print(f"{kind} r={r}: F1={m['Macro-F1']:.4f} MAE={m['MAE']:.4f}")

    out_dir = Path(args.save_path) if args.save_path else LOG_DIR
    out = out_dir / "robust_metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""附件3 推理 + 分类/回归头冲突统计（无标签，不报告准确率）。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import Att3Dataset  # noqa: E402
from models.tapir import TAPIR  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


def collate_att3(batch):
    names = [b.pop("name") for b in batch]
    out = {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}
    out["name"] = names
    return out


def analyze_conflicts(rows):
    """冲突定义（均基于模型双头输出，非金标）：
    1) neu_vs_reg_gt0.5: 分类=Neutral 且 |回归|>0.5
    2) neu_vs_reg_gt1.0: 分类=Neutral 且 |回归|>1.0
    3) sign_conflict: 分类为 Positive 但回归<0，或分类为 Negative 但回归>0
    """
    n = len(rows)
    neu_05 = neu_10 = sign = 0
    for r in rows:
        c = int(r["classification"])
        y = float(r["regression"])
        if c == 1 and abs(y) > 0.5:
            neu_05 += 1
        if c == 1 and abs(y) > 1.0:
            neu_10 += 1
        if (c == 2 and y < 0) or (c == 0 and y > 0):
            sign += 1
    return {
        "n": n,
        "neu_vs_reg_gt0.5": {"count": neu_05, "ratio": neu_05 / max(n, 1)},
        "neu_vs_reg_gt1.0": {"count": neu_10, "ratio": neu_10 / max(n, 1)},
        "sign_conflict": {"count": sign, "ratio": sign / max(n, 1)},
        "definition": {
            "neu_vs_reg_gt0.5": "classification==Neutral and |regression|>0.5",
            "neu_vs_reg_gt1.0": "classification==Neutral and |regression|>1.0",
            "sign_conflict": "Positive with regression<0, or Negative with regression>0",
        },
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(CKPT_DIR / "ours_selected_best.pt"),
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--bert_path", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=str(OUT_DIR / "att3_predictions.csv"))
    ap.add_argument("--out_json", type=str, default=str(OUT_DIR / "att3_conflict_stats.json"))
    args = ap.parse_args()

    ds = Att3Dataset()
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_att3)

    model = TAPIR(bert_path=args.bert_path)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in loader:
        names = batch.pop("name")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        out = model(batch)
        pred_c = int(out["logits"].argmax(-1).item())
        pred_r = float(out["y_hat"].item())
        unc = float(out["uncertainty"].item())
        neu05 = pred_c == 1 and abs(pred_r) > 0.5
        neu10 = pred_c == 1 and abs(pred_r) > 1.0
        sign_c = (pred_c == 2 and pred_r < 0) or (pred_c == 0 and pred_r > 0)
        rows.append(
            {
                "id": names[0],
                "classification": pred_c,
                "classification_name": CLS_NAME[pred_c],
                "regression": pred_r,
                "uncertainty": unc,
                "conflict_neu_gt0.5": int(neu05),
                "conflict_neu_gt1.0": int(neu10),
                "conflict_sign": int(sign_c),
            }
        )
        print(
            f"{names[0]}: cls={pred_c}({CLS_NAME[pred_c]}) reg={pred_r:.4f} "
            f"neu05={int(neu05)} neu10={int(neu10)} sign={int(sign_c)}"
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stats = analyze_conflicts(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved {args.out_csv} / {args.out_json}")


if __name__ == "__main__":
    main()

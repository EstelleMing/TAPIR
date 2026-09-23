"""问题二 Dataset：附件2 aligned 特征 + 训练期三视图缺失模拟。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import (
    TRAIN_RATES,
    apply_missing_to_inputs,
    detect_missing_masks,
    merge_masks,
    normalize_observed,
    reference_stats,
    sample_contiguous_mask,
)

ATT2_PKL = (
    Path(__file__).resolve().parents[3]
    / "E题数据"
    / "E题数据"
    / "附件2-数据集特征文件"
    / "aligned_50.pkl"
)
ATT3_DIR = (
    Path(__file__).resolve().parents[3]
    / "E题数据"
    / "E题数据"
    / "附件3-模态缺失特征样本"
    / "对齐版本"
)


def load_aligned_split(pkl_path: Path = ATT2_PKL) -> Dict[str, Dict[str, Any]]:
    import pickle

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


class MOSIAlignedDataset(Dataset):
    """CMU-MOSI aligned_50 特征。

    训练模式每个样本返回 full / M1 / M2 三视图；
    评测模式可注入外部人工掩码（与天然缺失合并）。
    """

    def __init__(
        self,
        split_data: Dict[str, Any],
        split_name: str,
        train_mode: bool = False,
        seed: int = 42,
        external_masks: Optional[Dict[str, np.ndarray]] = None,
        indices: Optional[List[int]] = None,
    ):
        self.split_name = split_name
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)
        self.external_masks = external_masks

        self.ids = list(split_data["id"])
        self.text_bert = np.asarray(split_data["text_bert"])
        self.audio = np.asarray(split_data["audio"], dtype=np.float32)
        self.vision = np.asarray(split_data["vision"], dtype=np.float32)
        self.y_cls = np.asarray(split_data["classification_labels"], dtype=np.int64)
        self.y_reg = np.asarray(split_data["regression_labels"], dtype=np.float32)
        self.indices = indices if indices is not None else list(range(len(self.ids)))

    def __len__(self) -> int:
        return len(self.indices)

    def _prepare_view(
        self,
        idx: int,
        art_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        tb0 = self.text_bert[idx].copy()
        a0 = self.audio[idx].copy()
        v0 = self.vision[idx].copy()

        Mt_n, Ma_n, Mv_n, L = detect_missing_masks(tb0, a0, v0)
        # 统计量只由天然观测确定，三个视图共用，避免遮挡改变 z-score
        stats = reference_stats(a0, v0, Ma_n, Mv_n, L)
        if art_masks is not None:
            Mt, Ma, Mv = merge_masks((Mt_n, Ma_n, Mv_n), art_masks)
            tb, a, v = apply_missing_to_inputs(tb0, a0, v0, art_masks[0], art_masks[1], art_masks[2])
            tb, a, v = apply_missing_to_inputs(tb, a, v, Mt_n, Ma_n, Mv_n)
        else:
            Mt, Ma, Mv = Mt_n, Ma_n, Mv_n
            tb, a, v = tb0, a0, v0

        a, v = normalize_observed(a, v, Ma, Mv, tb[1], L, stats=stats)
        # 人工遮挡位（相对完整视图）用于 NLL 监督
        if art_masks is not None:
            Art_t, Art_a, Art_v = art_masks
        else:
            Art_t = np.zeros_like(Mt)
            Art_a = np.zeros_like(Ma)
            Art_v = np.zeros_like(Mv)

        return {
            "input_ids": torch.tensor(tb[0], dtype=torch.long),
            "attention_mask": torch.tensor(tb[1], dtype=torch.long),
            "token_type_ids": torch.tensor(tb[2], dtype=torch.long),
            "audio": torch.tensor(a, dtype=torch.float32),
            "vision": torch.tensor(v, dtype=torch.float32),
            "Mt": torch.tensor(Mt, dtype=torch.float32),
            "Ma": torch.tensor(Ma, dtype=torch.float32),
            "Mv": torch.tensor(Mv, dtype=torch.float32),
            "Art_t": torch.tensor(Art_t, dtype=torch.float32),
            "Art_a": torch.tensor(Art_a, dtype=torch.float32),
            "Art_v": torch.tensor(Art_v, dtype=torch.float32),
            "Nat_t": torch.tensor(Mt_n, dtype=torch.float32),
            "Nat_a": torch.tensor(Ma_n, dtype=torch.float32),
            "Nat_v": torch.tensor(Mv_n, dtype=torch.float32),
            "L": torch.tensor(L, dtype=torch.long),
            "y_cls": torch.tensor(self.y_cls[idx], dtype=torch.long),
            "y_reg": torch.tensor(self.y_reg[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        T = self.text_bert.shape[-1]
        L = int(self.text_bert[idx, 1].sum())

        if self.train_mode:
            # full 视图：仅天然缺失
            full = self._prepare_view(idx, art_masks=None)
            views = {"full": full}
            for name in ("M1", "M2"):
                rate = float(self.rng.choice(TRAIN_RATES))
                art = sample_contiguous_mask(L, T, rate, self.rng, sync_modalities=None)
                views[name] = self._prepare_view(idx, art_masks=art)
            return views

        # 评测：可选外部掩码
        art = None
        if self.external_masks is not None:
            j = idx  # 外部掩码按原始 split 下标对齐
            art = (
                self.external_masks["Mt"][j],
                self.external_masks["Ma"][j],
                self.external_masks["Mv"][j],
            )
        return self._prepare_view(idx, art_masks=art)


def collate_eval(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def collate_train(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
    views = {}
    for vn in ("full", "M1", "M2"):
        views[vn] = collate_eval([b[vn] for b in batch])
    return views


class Att3Dataset(Dataset):
    """附件3：约 30 条无标签缺失样本。"""

    def __init__(self, att3_dir: Path = ATT3_DIR):
        import pickle

        self.samples = []
        paths = sorted(Path(att3_dir).glob("附件3_*.pkl"))
        for p in paths:
            with open(p, "rb") as f:
                d = pickle.load(f)["test"]
            self.samples.append(
                {
                    "name": p.stem,
                    "text_bert": np.asarray(d["text_bert"][0]),
                    "audio": np.asarray(d["audio"][0], dtype=np.float32),
                    "vision": np.asarray(d["vision"][0], dtype=np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.samples[i]
        tb0, a0, v0 = s["text_bert"], s["audio"], s["vision"]
        Mt, Ma, Mv, L = detect_missing_masks(tb0, a0, v0)
        a, v = normalize_observed(a0, v0, Ma, Mv, tb0[1], L)
        return {
            "name": s["name"],
            "input_ids": torch.tensor(tb0[0], dtype=torch.long),
            "attention_mask": torch.tensor(tb0[1], dtype=torch.long),
            "token_type_ids": torch.tensor(tb0[2], dtype=torch.long),
            "audio": torch.tensor(a, dtype=torch.float32),
            "vision": torch.tensor(v, dtype=torch.float32),
            "Mt": torch.tensor(Mt, dtype=torch.float32),
            "Ma": torch.tensor(Ma, dtype=torch.float32),
            "Mv": torch.tensor(Mv, dtype=torch.float32),
            "L": torch.tensor(L, dtype=torch.long),
        }


def class_weights_from_labels(y: np.ndarray, n_class: int = 3) -> torch.Tensor:
    """加权 CE：w_c = N / (3 * N_c)。"""
    y = np.asarray(y).astype(np.int64)
    N = len(y)
    counts = np.bincount(y, minlength=n_class).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = N / (n_class * counts)
    return torch.tensor(w, dtype=torch.float32)

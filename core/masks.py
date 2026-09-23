"""缺失掩码（missing mask）检测与连续块遮挡模拟。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 文本缺失占位：[UNK] token id = 100
TEXT_UNK_ID = 100
# 音/视 infinity-norm（无穷范数）阈值：低于视为缺失
EPS_MISS = 1e-6

# 训练遮挡率集合
TRAIN_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
# 连续块长度分布 P(1,2,3,4)
BLOCK_LEN_PROBS = np.array([0.65, 0.20, 0.10, 0.05], dtype=np.float64)
BLOCK_LENS = np.array([1, 2, 3, 4], dtype=np.int64)
# 模态组合：三模态同步 / 任意两模态 / 单模态
COMBO_PROBS = np.array([0.60, 0.25, 0.15], dtype=np.float64)


def effective_length(attention_mask: np.ndarray) -> int:
    """有效长度 L = sum(attention_mask)。"""
    return int(np.asarray(attention_mask).sum())


def content_slice(L: int) -> slice:
    """可判定缺失的内容位 t=1..L-2（排除 CLS、SEP、PAD）。"""
    if L < 3:
        return slice(0, 0)
    return slice(1, L - 1)


def detect_missing_masks(
    text_bert: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
    eps: float = EPS_MISS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """先检测缺失掩码。

    text_bert: (3, T) 或 (T,) input_ids 行；优先用 (3,T)。
    返回 Mt, Ma, Mv ∈ {0,1}^{T} 以及 L。仅在内容位标记缺失，其余为 0。
    """
    tb = np.asarray(text_bert)
    if tb.ndim == 2 and tb.shape[0] == 3:
        input_ids = tb[0].astype(np.int64)
        attn = tb[1]
    elif tb.ndim == 1:
        input_ids = tb.astype(np.int64)
        attn = (input_ids != 0).astype(np.float64)
    else:
        raise ValueError(f"unexpected text_bert shape: {tb.shape}")

    T = input_ids.shape[0]
    L = effective_length(attn)
    sl = content_slice(L)

    Mt = np.zeros(T, dtype=np.float32)
    Ma = np.zeros(T, dtype=np.float32)
    Mv = np.zeros(T, dtype=np.float32)

    if sl.start == sl.stop:
        return Mt, Ma, Mv, L

    Mt[sl] = (input_ids[sl] == TEXT_UNK_ID).astype(np.float32)

    a = np.asarray(audio, dtype=np.float64)
    v = np.asarray(vision, dtype=np.float64)
    a_inf = np.max(np.abs(a), axis=-1)
    v_inf = np.max(np.abs(v), axis=-1)
    Ma[sl] = (a_inf[sl] < eps).astype(np.float32)
    Mv[sl] = (v_inf[sl] < eps).astype(np.float32)
    return Mt, Ma, Mv, L


def _axis_stats(x: np.ndarray, mask: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """沿观测时间步估计每个特征维的均值和标准差。"""
    dim = x.shape[-1]
    if not mask.any():
        return np.zeros((1, dim), dtype=np.float32), np.ones((1, dim), dtype=np.float32)
    mu = x[mask].mean(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(x[mask].std(axis=0, keepdims=True), eps).astype(np.float32)
    return mu, sd


def reference_stats(
    audio: np.ndarray,
    vision: np.ndarray,
    Ma_nat: np.ndarray,
    Mv_nat: np.ndarray,
    L: int,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用天然观测位估计 z-score 统计量，供 full / M1 / M2 共用。"""
    a = np.asarray(audio, dtype=np.float32)
    v = np.asarray(vision, dtype=np.float32)
    valid = np.zeros(a.shape[0], dtype=bool)
    if L > 0:
        valid[:L] = True
    mu_a, sd_a = _axis_stats(a, (Ma_nat < 0.5) & valid, eps)
    mu_v, sd_v = _axis_stats(v, (Mv_nat < 0.5) & valid, eps)
    return mu_a, sd_a, mu_v, sd_v


def normalize_observed(
    audio: np.ndarray,
    vision: np.ndarray,
    Ma: np.ndarray,
    Mv: np.ndarray,
    attn: np.ndarray,
    L: int,
    eps: float = 1e-5,
    stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """用给定统计量变换当前仍观测的位置；缺失位保持 0。

    stats 来自完整视图的天然观测。不传时退回用当前掩码估计，仅作兼容。
    """
    a = np.asarray(audio, dtype=np.float32).copy()
    v = np.asarray(vision, dtype=np.float32).copy()
    if L < a.shape[0]:
        a[L:] = 0.0
        v[L:] = 0.0

    valid = np.zeros(a.shape[0], dtype=bool)
    if L > 0:
        valid[:L] = True
    mask_a = ((1.0 - Ma) > 0.5) & valid
    mask_v = ((1.0 - Mv) > 0.5) & valid
    if stats is None:
        stats = reference_stats(a, v, Ma, Mv, L, eps)
    mu_a, sd_a, mu_v, sd_v = stats

    if mask_a.any():
        a[mask_a] = (a[mask_a] - mu_a) / sd_a
    a[~mask_a] = 0.0
    if mask_v.any():
        v[mask_v] = (v[mask_v] - mu_v) / sd_v
    v[~mask_v] = 0.0
    return a, v


def _sample_modality_combo(rng: np.random.Generator) -> List[str]:
    """按概率采样参与遮挡的模态集合。"""
    choice = rng.choice(3, p=COMBO_PROBS)
    mods = ["t", "a", "v"]
    if choice == 0:
        return mods  # 三模态同步
    if choice == 1:
        return list(rng.choice(mods, size=2, replace=False))
    return [str(rng.choice(mods))]


def sample_contiguous_mask(
    L: int,
    T: int,
    rate: float,
    rng: np.random.Generator,
    sync_modalities: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在内容位上采样连续块缺失掩码。

    rate=0 时返回全零。sync_modalities 非空则这些模态共用同一时间块。
    """
    Mt = np.zeros(T, dtype=np.float32)
    Ma = np.zeros(T, dtype=np.float32)
    Mv = np.zeros(T, dtype=np.float32)
    N = max(L - 2, 0)
    if rate <= 0 or N <= 0:
        return Mt, Ma, Mv

    K = max(1, int(round(rate * N)))
    covered = np.zeros(N, dtype=bool)
    positions: List[int] = []
    guard = 0
    while len(positions) < K and guard < 10000:
        guard += 1
        plen = int(rng.choice(BLOCK_LENS, p=BLOCK_LEN_PROBS))
        plen = min(plen, N)
        start = int(rng.integers(0, N - plen + 1))
        for i in range(start, start + plen):
            if not covered[i]:
                covered[i] = True
                positions.append(i)
                if len(positions) >= K:
                    break
    # 截断到 K
    positions = positions[:K]
    # 内容位索引 = positions + 1
    content_idx = np.array(positions, dtype=np.int64) + 1

    if sync_modalities is None:
        mods = _sample_modality_combo(rng)
    else:
        mods = list(sync_modalities)

    if "t" in mods:
        Mt[content_idx] = 1.0
    if "a" in mods:
        Ma[content_idx] = 1.0
    if "v" in mods:
        Mv[content_idx] = 1.0
    return Mt, Ma, Mv


def apply_missing_to_inputs(
    text_bert: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
    Mt: np.ndarray,
    Ma: np.ndarray,
    Mv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """应用人工遮挡：文本填 100，音视频填 0。保留天然缺失。"""
    tb = np.asarray(text_bert).copy()
    a = np.asarray(audio, dtype=np.float32).copy()
    v = np.asarray(vision, dtype=np.float32).copy()
    if tb.ndim == 2 and tb.shape[0] == 3:
        ids = tb[0].astype(np.int64)
        ids[Mt > 0.5] = TEXT_UNK_ID
        tb[0] = ids
    else:
        raise ValueError("text_bert must be (3,T)")
    a[Ma > 0.5] = 0.0
    v[Mv > 0.5] = 0.0
    return tb, a, v


def merge_masks(
    nat: Tuple[np.ndarray, np.ndarray, np.ndarray],
    art: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """天然缺失 ∪ 人工遮挡。"""
    return (
        np.maximum(nat[0], art[0]).astype(np.float32),
        np.maximum(nat[1], art[1]).astype(np.float32),
        np.maximum(nat[2], art[2]).astype(np.float32),
    )


def generate_eval_masks(
    text_berts: np.ndarray,
    rates: Sequence[float],
    seed: int,
    out_dir: Path,
    sync_modalities: Sequence[str] = ("t", "a", "v"),
) -> Dict[float, Dict[str, np.ndarray]]:
    """为整个 split 预生成固定缺失掩码（三模态同步连续块）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N, _, T = text_berts.shape
    result: Dict[float, Dict[str, np.ndarray]] = {}
    for r in rates:
        rng = np.random.default_rng(seed + int(r * 1000))
        Mt = np.zeros((N, T), dtype=np.float32)
        Ma = np.zeros((N, T), dtype=np.float32)
        Mv = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            L = effective_length(text_berts[i, 1])
            mt, ma, mv = sample_contiguous_mask(
                L, T, float(r), rng, sync_modalities=sync_modalities
            )
            Mt[i], Ma[i], Mv[i] = mt, ma, mv
        pack = {"Mt": Mt, "Ma": Ma, "Mv": Mv, "rate": np.array([r], dtype=np.float32)}
        path = out_dir / f"test_mask_r{r:.1f}_seed{seed}.npz"
        np.savez_compressed(path, **pack)
        result[float(r)] = pack
        print(f"[masks] saved {path}  mean_Mt={Mt.mean():.4f}")
    return result


def load_eval_masks(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}

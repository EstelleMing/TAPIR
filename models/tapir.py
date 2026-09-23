"""TAPIR（Temporal missing-Aware Probabilistic Imputation with Reliability gating，时序缺失感知的概率补全与可靠性门控）与 B0 基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

DEFAULT_BERT = Path("/home/yangming/Huawei-E/E题/model/bert-base-uncased")


def resolve_bert_path(path: Optional[str] = None) -> str:
    if path:
        return path
    if DEFAULT_BERT.exists():
        return str(DEFAULT_BERT)
    return "bert-base-uncased"


class TemporalAttnPool(nn.Module):
    """时序注意力池化（temporal attention pooling）。"""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B,T,d)  mask: (B,T) 1=valid
        logits = self.score(h).squeeze(-1)  # (B,T)
        logits = logits.masked_fill(mask <= 0, -1e9)
        w = torch.softmax(logits, dim=-1)
        return torch.sum(h * w.unsqueeze(-1), dim=1)


class FusionTransformer(nn.Module):
    """2 层双向 Transformer；缺失位可作 Query，不作 Key/Value（observed-key-mask）。"""

    def __init__(self, d: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # 关闭 nested tensor，避免 key_padding_mask 改变输出长度
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=nlayers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d)
        observed_mask: (B, S) 1=观测(可作 K/V)，0=缺失(不作 K/V)
        """
        # src_key_padding_mask: True 表示忽略该 key（须为 bool）
        key_pad = (observed_mask <= 0.5).to(dtype=torch.bool)
        # 若整行全被 mask，强制至少一个 key，避免 NaN
        all_miss = key_pad.all(dim=-1)
        if all_miss.any():
            key_pad = key_pad.clone()
            key_pad[all_miss, 0] = False
        return self.encoder(x, src_key_padding_mask=key_pad)


class MultimodalEncoder(nn.Module):
    """三模态投影 + 位置/模态/缺失嵌入 + Transformer。"""

    def __init__(
        self,
        d: int = 128,
        audio_dim: int = 74,
        vision_dim: int = 35,
        max_len: int = 50,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.2,
        bert_path: Optional[str] = None,
    ):
        super().__init__()
        self.d = d
        self.bert = BertModel.from_pretrained(resolve_bert_path(bert_path))
        self.text_proj = nn.Linear(self.bert.config.hidden_size, d)
        self.audio_proj = nn.Linear(audio_dim, d)
        self.vision_proj = nn.Linear(vision_dim, d)

        self.pos_emb = nn.Embedding(max_len, d)
        self.mod_emb = nn.Embedding(3, d)  # 0=text,1=audio,2=vision
        self.miss_emb = nn.Embedding(2, d)  # 0=obs,1=miss
        self.dropout = nn.Dropout(dropout)
        self.fusion = FusionTransformer(d, nhead, nlayers, dropout)

    def encode_modalities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        Mt: torch.Tensor,
        Ma: torch.Tensor,
        Mv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 Ht,Ha,Hv 与时间有效 mask（B,T）。"""
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        Ht = self.text_proj(bert_out.last_hidden_state)  # (B,T,d)
        Ha = self.audio_proj(audio)
        Hv = self.vision_proj(vision)

        B, T, _ = Ht.shape
        device = Ht.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pe = self.pos_emb(pos)

        def add_emb(H, M, mid):
            me = self.mod_emb(torch.full((B, T), mid, device=device, dtype=torch.long))
            miss = self.miss_emb(M.long().clamp(0, 1))
            return self.dropout(H + pe + me + miss)

        Ht = add_emb(Ht, Mt, 0)
        Ha = add_emb(Ha, Ma, 1)
        Hv = add_emb(Hv, Mv, 2)
        time_mask = attention_mask.float()
        return Ht, Ha, Hv, time_mask

    def fuse_sequence(
        self,
        Ht: torch.Tensor,
        Ha: torch.Tensor,
        Hv: torch.Tensor,
        Mt: torch.Tensor,
        Ma: torch.Tensor,
        Mv: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """交错拼接为 (B, 3T, d)，observed-key-mask 融合。"""
        B, T, d = Ht.shape
        x = torch.stack([Ht, Ha, Hv], dim=2).reshape(B, 3 * T, d)
        # 观测 = 未缺失 且 在有效长度内
        obs_t = (1.0 - Mt) * time_mask
        obs_a = (1.0 - Ma) * time_mask
        obs_v = (1.0 - Mv) * time_mask
        obs = torch.stack([obs_t, obs_a, obs_v], dim=2).reshape(B, 3 * T)
        h = self.fusion(x, obs)
        # 还原各模态
        h = h.view(B, T, 3, d)
        return h, time_mask


class ReconHead(nn.Module):
    """概率补全：预测 μ 与 σ²=softplus+ε。"""

    def __init__(self, d: int = 128, eps: float = 1e-4):
        super().__init__()
        self.mu = nn.Linear(d, d)
        self.raw_var = nn.Linear(d, d)
        self.eps = eps

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(h)
        var = F.softplus(self.raw_var(h)) + self.eps
        return mu, var


class TAPIR(nn.Module):
    """时序缺失感知的概率补全与可靠性门控，外加双任务头。"""

    def __init__(
        self,
        d: int = 128,
        gamma: float = 1.0,
        bert_path: Optional[str] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gamma = gamma
        self.encoder = MultimodalEncoder(d=d, dropout=dropout, bert_path=bert_path)
        self.recon_t = ReconHead(d)
        self.recon_a = ReconHead(d)
        self.recon_v = ReconHead(d)
        self.pool = TemporalAttnPool(d)
        self.cls_head = nn.Linear(d, 3)
        self.reg_head = nn.Linear(d, 1)

    def set_bert_trainable(self, mode: str) -> None:
        """mode: freeze | last2 | all"""
        bert = self.encoder.bert
        for p in bert.parameters():
            p.requires_grad = False
        if mode == "freeze":
            return
        if mode == "last2":
            for layer in bert.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in bert.pooler.parameters():
                p.requires_grad = True
            return
        if mode == "all":
            for p in bert.parameters():
                p.requires_grad = True

    def forward(self, batch: Dict[str, torch.Tensor], return_latent: bool = True) -> Dict[str, torch.Tensor]:
        Ht0, Ha0, Hv0, time_mask = self.encoder.encode_modalities(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["audio"],
            batch["vision"],
            batch["Mt"],
            batch["Ma"],
            batch["Mv"],
        )
        # 融合前的模态表示作为完整视图 latent 目标（对 full 视图）
        # 先过 Transformer 得到上下文，再做概率补全
        h_stack, _ = self.encoder.fuse_sequence(
            Ht0, Ha0, Hv0, batch["Mt"], batch["Ma"], batch["Mv"], time_mask
        )
        Ht, Ha, Hv = h_stack[:, :, 0], h_stack[:, :, 1], h_stack[:, :, 2]

        mu_t, var_t = self.recon_t(Ht)
        mu_a, var_a = self.recon_a(Ha)
        mu_v, var_v = self.recon_v(Hv)

        Mt, Ma, Mv = batch["Mt"], batch["Ma"], batch["Mv"]
        # Z = (1-M)H + M*μ ；只替换缺失位
        Zt = (1 - Mt).unsqueeze(-1) * Ht + Mt.unsqueeze(-1) * mu_t
        Za = (1 - Ma).unsqueeze(-1) * Ha + Ma.unsqueeze(-1) * mu_a
        Zv = (1 - Mv).unsqueeze(-1) * Hv + Mv.unsqueeze(-1) * mu_v

        # 可靠性由预测方差 σ² 决定：观测位 r=1，缺失位 r=exp(-γ σ²)
        u_t = var_t.mean(dim=-1)
        u_a = var_a.mean(dim=-1)
        u_v = var_v.mean(dim=-1)
        r_t = (1 - Mt) + Mt * torch.exp(-self.gamma * u_t)
        r_a = (1 - Ma) + Ma * torch.exp(-self.gamma * u_a)
        r_v = (1 - Mv) + Mv * torch.exp(-self.gamma * u_v)

        # 按可靠性归一化。r→0 时权重→0，避免 softmax 在 [0,1] 上的下限
        R = torch.stack([r_t, r_a, r_v], dim=-1) * time_mask.unsqueeze(-1)
        alpha = R / R.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        Z = (
            alpha[:, :, 0:1] * Zt
            + alpha[:, :, 1:2] * Za
            + alpha[:, :, 2:3] * Zv
        )
        h = self.pool(Z, time_mask)
        logits = self.cls_head(h)
        y_hat = 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))

        # 分母按“缺失的模态×位置”计数，与分子一致；完全无缺失时为 0
        numer = (var_t.mean(-1) * Mt + var_a.mean(-1) * Ma + var_v.mean(-1) * Mv) * time_mask
        den = (Mt + Ma + Mv) * time_mask
        numer_s = numer.sum(-1)
        den_s = den.sum(-1)
        unc = torch.where(den_s > 0, numer_s / den_s, torch.zeros_like(numer_s))

        out = {
            "logits": logits,
            "y_hat": y_hat,
            "uncertainty": unc,
            "Zt": Zt,
            "Za": Za,
            "Zv": Zv,
            "mu_t": mu_t,
            "mu_a": mu_a,
            "mu_v": mu_v,
            "var_t": var_t,
            "var_a": var_a,
            "var_v": var_v,
            "Ht_ctx": Ht,
            "Ha_ctx": Ha,
            "Hv_ctx": Hv,
            "Ht0": Ht0,
            "Ha0": Ha0,
            "Hv0": Hv0,
            "alpha": alpha,
            "time_mask": time_mask,
        }
        return out


class BaselineB0(nn.Module):
    """B0：零填充 + 同结构融合 Transformer，无概率补全 / 无蒸馏。"""

    def __init__(self, d: int = 128, bert_path: Optional[str] = None, dropout: float = 0.2):
        super().__init__()
        self.encoder = MultimodalEncoder(d=d, dropout=dropout, bert_path=bert_path)
        self.pool = TemporalAttnPool(d)
        self.cls_head = nn.Linear(d, 3)
        self.reg_head = nn.Linear(d, 1)

    def set_bert_trainable(self, mode: str) -> None:
        bert = self.encoder.bert
        for p in bert.parameters():
            p.requires_grad = False
        if mode == "freeze":
            return
        if mode == "last2":
            for layer in bert.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in bert.pooler.parameters():
                p.requires_grad = True
            return
        if mode == "all":
            for p in bert.parameters():
                p.requires_grad = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Ht, Ha, Hv, time_mask = self.encoder.encode_modalities(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["audio"],
            batch["vision"],
            batch["Mt"],
            batch["Ma"],
            batch["Mv"],
        )
        h_stack, _ = self.encoder.fuse_sequence(
            Ht, Ha, Hv, batch["Mt"], batch["Ma"], batch["Mv"], time_mask
        )
        # 简单均值融合三模态
        Z = h_stack.mean(dim=2)
        h = self.pool(Z, time_mask)
        logits = self.cls_head(h)
        y_hat = 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))
        return {"logits": logits, "y_hat": y_hat, "uncertainty": torch.zeros(h.size(0), device=h.device)}

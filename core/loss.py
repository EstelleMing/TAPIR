"""问题二损失：加权 CE + Huber + Corr + Gaussian NLL + 蒸馏 + 一致性；L_elastic 接口默认 0。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    abs_e = err.abs()
    quad = torch.clamp(abs_e, max=delta)
    lin = abs_e - quad
    return (0.5 * quad ** 2 + delta * lin).mean()


def corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """λ(1 - Corr)；返回 (1-Corr) 以便外部乘 λ。"""
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = torch.sqrt((p ** 2).sum() * (t ** 2).sum()).clamp(min=eps)
    corr = num / den
    return 1.0 - corr


def gaussian_nll(
    mu: torch.Tensor,
    var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """仅在人工遮挡位：0.5*(log var + (x-μ)^2 / var)。mask: (B,T)。"""
    # mu/var/target: (B,T,d)
    m = mask.unsqueeze(-1)
    n = m.sum() * mu.size(-1)
    if n <= 0:
        return mu.new_zeros(())
    nll = 0.5 * (torch.log(var.clamp(min=eps)) + (target - mu) ** 2 / var.clamp(min=eps))
    return (nll * m).sum() / n.clamp(min=1.0)


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """置信度可加权的 JS divergence（Jensen–Shannon 散度）。返回 per-sample JS。"""
    p = F.softmax(p_logits, dim=-1).clamp(min=eps)
    q = F.softmax(q_logits, dim=-1).clamp(min=eps)
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (torch.log(p) - torch.log(m))).sum(-1)
        + (q * (torch.log(q) - torch.log(m))).sum(-1)
    )
    return js


def prediction_confidence(logits: torch.Tensor) -> torch.Tensor:
    """用 max softmax 作为置信度权重。"""
    return F.softmax(logits, dim=-1).max(dim=-1).values


class Problem2Criterion(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        lambda_corr: float = 0.5,
        w_nll: float = 0.5,
        w_distill: float = 0.5,
        w_cons: float = 0.2,
        w_elastic: float = 0.0,  # L_elastic 接口，默认关闭
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lambda_corr = lambda_corr
        self.w_nll = w_nll
        self.w_distill = w_distill
        self.w_cons = w_cons
        self.w_elastic = w_elastic
        self.huber_delta = huber_delta

    def task_loss(
        self, logits: torch.Tensor, y_hat: torch.Tensor, y_cls: torch.Tensor, y_reg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = F.cross_entropy(logits, y_cls, weight=self.class_weights)
        hub = huber_loss(y_hat, y_reg, self.huber_delta)
        corr_term = corr_loss(y_hat, y_reg)
        loss = ce + hub + self.lambda_corr * corr_term
        return loss, {
            "ce": float(ce.detach()),
            "huber": float(hub.detach()),
            "corr_term": float(corr_term.detach()),
        }

    def elastic_loss(self, *args, **kwargs) -> torch.Tensor:
        """L_elastic 占位；权重默认 0。"""
        return torch.zeros((), device=self.class_weights.device)

    def forward(
        self,
        out_full: Dict[str, torch.Tensor],
        out_m1: Dict[str, torch.Tensor],
        out_m2: Dict[str, torch.Tensor],
        batch_full: Dict[str, torch.Tensor],
        batch_m1: Dict[str, torch.Tensor],
        batch_m2: Dict[str, torch.Tensor],
        teacher_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        y_cls = batch_full["y_cls"]
        y_reg = batch_full["y_reg"]

        # 双任务：三视图均监督
        loss_f, d_f = self.task_loss(out_full["logits"], out_full["y_hat"], y_cls, y_reg)
        loss_1, d_1 = self.task_loss(out_m1["logits"], out_m1["y_hat"], y_cls, y_reg)
        loss_2, d_2 = self.task_loss(out_m2["logits"], out_m2["y_hat"], y_cls, y_reg)
        task = (loss_f + loss_1 + loss_2) / 3.0

        # Gaussian NLL：缺失视图的人工遮挡位，目标为完整视图 latent（stop-grad）
        # 用 full 视图的 Ht0/Ha0/Hv0（编码后、补全前）作为目标
        tgt_t = out_full["Ht0"].detach()
        tgt_a = out_full["Ha0"].detach()
        tgt_v = out_full["Hv0"].detach()

        def sup(batch: Dict[str, torch.Tensor], modal: str) -> torch.Tensor:
            # 只监督“人工遮挡且原本有观测”的位置
            return batch[f"Art_{modal}"] * (1.0 - batch[f"Nat_{modal}"])

        nll = out_m1["mu_t"].new_zeros(())
        nll = nll + gaussian_nll(out_m1["mu_t"], out_m1["var_t"], tgt_t, sup(batch_m1, "t"))
        nll = nll + gaussian_nll(out_m1["mu_a"], out_m1["var_a"], tgt_a, sup(batch_m1, "a"))
        nll = nll + gaussian_nll(out_m1["mu_v"], out_m1["var_v"], tgt_v, sup(batch_m1, "v"))
        nll = nll + gaussian_nll(out_m2["mu_t"], out_m2["var_t"], tgt_t, sup(batch_m2, "t"))
        nll = nll + gaussian_nll(out_m2["mu_a"], out_m2["var_a"], tgt_a, sup(batch_m2, "a"))
        nll = nll + gaussian_nll(out_m2["mu_v"], out_m2["var_v"], tgt_v, sup(batch_m2, "v"))
        nll = nll / 6.0

        # 蒸馏：教师优先 EMA full；置信度加权 JS + |y_miss - sg(y_full)|
        if teacher_out is not None:
            t_logits = teacher_out["logits"].detach()
            t_y = teacher_out["y_hat"].detach()
        else:
            t_logits = out_full["logits"].detach()
            t_y = out_full["y_hat"].detach()

        conf = prediction_confidence(t_logits).detach()
        js1 = js_divergence(out_m1["logits"], t_logits)
        js2 = js_divergence(out_m2["logits"], t_logits)
        reg1 = (out_m1["y_hat"] - t_y).abs()
        reg2 = (out_m2["y_hat"] - t_y).abs()
        distill = ((conf * (js1 + reg1)).mean() + (conf * (js2 + reg2)).mean()) / 2.0

        # 两缺失视图一致性
        js12 = js_divergence(out_m1["logits"], out_m2["logits"]).mean()
        reg12 = (out_m1["y_hat"] - out_m2["y_hat"]).abs().mean()
        cons = js12 + reg12

        elastic = self.elastic_loss()

        total = (
            task
            + self.w_nll * nll
            + self.w_distill * distill
            + self.w_cons * cons
            + self.w_elastic * elastic
        )
        stats = {
            "loss": float(total.detach()),
            "task": float(task.detach()),
            "nll": float(nll.detach() if torch.is_tensor(nll) else nll),
            "distill": float(distill.detach()),
            "cons": float(cons.detach()),
            "ce": (d_f["ce"] + d_1["ce"] + d_2["ce"]) / 3.0,
            "huber": (d_f["huber"] + d_1["huber"] + d_2["huber"]) / 3.0,
        }
        return total, stats


class BaselineCriterion(nn.Module):
    """B0：仅加权 CE + Huber + Corr。"""

    def __init__(self, class_weights: torch.Tensor, lambda_corr: float = 0.5, huber_delta: float = 1.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lambda_corr = lambda_corr
        self.huber_delta = huber_delta

    def forward(
        self, out: Dict[str, torch.Tensor], y_cls: torch.Tensor, y_reg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = F.cross_entropy(out["logits"], y_cls, weight=self.class_weights)
        hub = huber_loss(out["y_hat"], y_reg, self.huber_delta)
        corr_term = corr_loss(out["y_hat"], y_reg)
        total = ce + hub + self.lambda_corr * corr_term
        return total, {
            "loss": float(total.detach()),
            "ce": float(ce.detach()),
            "huber": float(hub.detach()),
            "corr_term": float(corr_term.detach()),
        }

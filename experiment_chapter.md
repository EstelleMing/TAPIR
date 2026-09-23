# 问题二实验章节：TAPIR（延长训练诊断）

> **表述约定**：本方法称为 **TAPIR**（Temporal missing-Aware Probabilistic Imputation with Reliability gating，时序缺失感知的概率补全与可靠性门控）。结构受 EASE（Elastic Asynchronous Sequence Encoding，弹性异步序列编码）启发，但本版 **未启用** \(L_{elastic}\)，**不能**声称已验证弹性对齐，也不等同于官方 EASE。此前相对 B0 的 clean/缺失优势仅表明**完整模型**优于零填充基线，**不能**单独证明概率补全、可靠性门控、蒸馏三者各自有效——模块消融留待后续。
>
> **本章性质**：延长训练 + BERT 冻结日程对比 + 逐类混淆诊断；**不是**模块消融。数字全部来自 `logs/ours_freeze*_history.*`、`logs/ours_extended_summary.json`、`outputs/att3_conflict_stats.json` 实跑结果。

## 1. 方法简述（结构未改）

词级对齐 MSA；文本经本地 `bert-base-uncased`；音/视投影至 128；位置+模态+缺失嵌入；2 层双向 Transformer（observed-key-mask）；缺失位预测 \(\mu,\sigma^2\) 并做可靠性门控融合；分类 softmax + 回归 \(3\tanh\)。损失含加权 CE、Huber、Corr 项、Gaussian NLL、EMA 蒸馏与双缺失一致性；**\(w_{elastic}=0\)**。

## 2. 本次实验设置

| 项目 | 设置 |
|------|------|
| 目的 | 诊断是否欠拟合；对比冻结日程；输出逐类混淆 |
| 数据 | 附件2 `aligned_50.pkl`；valid 选模，**test 仅最终评一次** |
| 选模指标 | \(0.5\times\mathrm{Acc}+0.5\times\mathrm{Macro\text{-}F1}\)（仅 valid） |
| 预算 | 上限 40 epoch，patience=8；batch=64；seed=42；RTX 3090 |
| 对比 | **冻结前 2 轮** vs **冻结前 5 轮**（之后均只解冻 BERT 最后两层）；各自从零训练，公平对比 |
| 缺失探针 | valid 上固定掩码（seed=2026+offset，三模态同步连续块）rate∈{0.3,0.5}，每轮评 Macro-F1/MAE，**不每轮重随机** |

脚本：`train_extended.py`；日志：`logs/ours_freeze2_history.csv|json`、`ours_freeze5_history.csv|json`。

## 3. 冻结 2 vs 冻结 5（valid 选模）

**表 1. 两种冻结日程的 valid 最佳点**

| 日程 | best epoch | valid Acc | valid Macro-F1 | 综合分 | 实际跑了 | early stop |
|------|------------|-----------|----------------|--------|----------|------------|
| freeze2 | **13** | 0.6346 | 0.6154 | 0.6250 | 21 | 是（patience=8） |
| freeze5 | **11** | **0.6374** | **0.6155** | **0.6264** | 19 | 是 |

**选定配置**：`freeze5`（综合分略高）。test **仅对此配置评一次**。

### 3.1 曲线摘要

- **freeze2**：综合分在 ep13 达峰（0.6250），此后至 ep21 未再刷新；ep19 曾回升至 0.6224，仍低于 best。**best 之后未持续上升**。
- **freeze5**：冻结段（ep≤5）综合分约 0.38→0.54；解冻后 ep11 达综合分峰值 0.6264；ep13 Macro-F1 略高（0.6187）但综合分 0.6253 略低故未选。**best 之后未持续上升**，ep19 综合分 0.6139。
- 相对此前 8 epoch 轻量跑（valid Macro-F1≈0.595），延长训练把 valid Macro-F1 提到约 **0.615**，有收益但**仍停在 ~0.60 一带**，未见冲向 0.75 的趋势。

## 4. 选定配置：逐类 P/R/F1 与混淆矩阵

### 4.1 Valid（freeze5，epoch 11）

| 类 | Precision | Recall | F1 | support |
|----|-----------|--------|-----|---------|
| Negative | 0.6749 | 0.6650 | 0.6699 | 206 |
| Neutral | **0.4574** | **0.4674** | **0.4624** | 184 |
| Positive | 0.7151 | 0.7130 | 0.7141 | 338 |
| **Overall** | Acc=**0.6374** | Macro-F1=**0.6155** | MAE=0.6912 | Corr=0.6598 |

混淆矩阵（行=真值 Neg/Neu/Pos，列=预测）：

```
          pred Neg  Neu  Pos
true Neg     137    37   32
true Neu      34    86   64
true Pos      32    65  241
```

解读：Neutral 大量漏到 Positive（64）与 Negative（34）；Positive↔Neutral 互换也多（65）。**Neutral 是主瓶颈**。

### 4.2 Test（唯一一次，同权重）

| 类 | Precision | Recall | F1 | support |
|----|-----------|--------|-----|---------|
| Negative | 0.6941 | 0.7343 | 0.7136 | 207 |
| Neutral | **0.4497** | **0.4241** | **0.4365** | 158 |
| Positive | 0.7772 | 0.7707 | 0.7739 | 362 |
| **Overall** | Acc=**0.6850** | Macro-F1=**0.6413** | MAE=0.7116 | Corr=0.6984 |

混淆矩阵：

```
          pred Neg  Neu  Pos
true Neg     152    33   22
true Neu      33    67   58
true Pos      34    49  279
```

Test Acc/Macro-F1 相对 valid 略高，但 Neutral F1 仍约 0.44，与「整体 >0.75」目标差距主要来自中性类。

## 5. Valid 缺失探针（选定配置 best 轮）

固定掩码下（非每轮重采样）：

| rate | Macro-F1 | MAE |
|------|----------|-----|
| 0.3 | 0.5557 | 0.7671 |
| 0.5 | 0.5363 | 0.8241 |

相对 clean valid Macro-F1=0.6155，缺失 0.3/0.5 分别下降约 6.0 / 7.9 个百分点。

（对照：freeze2 在其 best 轮 miss0.3/0.5 的 Macro-F1 为 0.5470 / 0.5435。）

## 6. 附件3：双头冲突（无准确率声称）

权重：`checkpoints/ours_selected_best.pt`（freeze5）。输出：`outputs/att3_predictions.csv`、`outputs/att3_conflict_stats.json`。

**冲突定义**（仅模型双头自洽性，**无金标**）：

1. `neu_vs_reg_gt0.5`：分类=Neutral 且 \(|y|>0.5\)
2. `neu_vs_reg_gt1.0`：分类=Neutral 且 \(|y|>1.0\)
3. `sign_conflict`：分类 Positive 但 \(y<0\)，或 Negative 但 \(y>0\)

| 冲突类型 | 条数 / 30 | 比例 |
|----------|-----------|------|
| Neutral 且 \|y\|>0.5 | 3 | **10.0%** |
| Neutral 且 \|y\|>1.0 | 0 | **0.0%** |
| 符号冲突 | 0 | **0.0%** |

不报告附件3「准确率」；冲突比例仅作双任务一致性诊断。

## 7. 结论与下一步

1. 延长训练有效但有限：valid Macro-F1 从 ~0.60 提到 ~0.615，综合分 ~0.626 后**平台**，early stop 合理；**未见冲向 0.75 的上升通道**。
2. freeze2 与 freeze5 几乎打平（0.6250 vs 0.6264）；多解冻几轮 BERT 并非当前主矛盾。
3. **优先排查数据处理与 Neutral 决策**（边界样本、标签映射、决策阈值、类不平衡与混淆方向），而不是先做三模块消融或拧损失权重追点。
4. 若后续 Neutral F1 与整体 Macro-F1 **明显**突破当前平台（例如稳定 >0.70），再开展结构消融（补全 / 门控 / 蒸馏）以支撑论文因果表述。
5. 完整模型相对 B0 的优势仍可作为对照叙述，但须写明**非单模块证明**。

## 8. 复现

```bash
cd /home/yangming/Huawei-E/E题/E_solution/problem2
/home/yangming/anaconda3/envs/pai/bin/python train_extended.py \
  --epochs 40 --patience 8 --batch_size 64 --seed 42 --freeze_list 2 5 --device cuda:0
/home/yangming/anaconda3/envs/pai/bin/python infer_att3.py \
  --ckpt checkpoints/ours_selected_best.pt --device cuda:0
```

# 问题二：TAPIR

**TAPIR**（Temporal missing-Aware Probabilistic Imputation with Reliability gating，时序缺失感知的概率补全与可靠性门控）。

数据为附件2 `aligned_50.pkl`，`L_elastic` 默认权重为 0。

## 目录

```
configs/          训练 / 评测 YAML
core/             dataset、masks、loss、utils
models/           tapir.py（TAPIR / BaselineB0）
train.py          轻量训练（TAPIR + B0）
train_extended.py 延长训练与冻结日程
train.sh
eval_robust.py    缺失鲁棒评测
robust_eval.sh
infer_att3.py     附件3 推理
calibrate_regression.py
checkpoints/  logs/  masks/  outputs/
```

## 环境

- Python：`/home/yangming/anaconda3/envs/pai/bin/python`
- 依赖见 `requirements.txt`
- BERT：`/home/yangming/Huawei-E/E题/model/bert-base-uncased`

## 数据

| 数据 | 路径 |
|------|------|
| 附件2 aligned | `E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl` |
| 附件3 对齐版 | `E题数据/E题数据/附件3-模态缺失特征样本/对齐版本` |

读取位置：`core/dataset.py` 中的 `ATT2_PKL`、`ATT3_DIR`。

## 训练

```bash
cd /home/yangming/Huawei-E/E题/E_solution/problem2
sh train.sh
# 或
/home/yangming/anaconda3/envs/pai/bin/python train_extended.py --config_file configs/train_mosi.yaml
```

轻量对照（最多 8 epoch）：

```bash
/home/yangming/anaconda3/envs/pai/bin/python train.py \
  --model both --epochs 8 --patience 3 --seed 42 --batch_size 64 --device cuda:0
```

命令行里写明的参数会覆盖 YAML 同名字段。

## 鲁棒评测

```bash
sh robust_eval.sh
```

## 附件3

```bash
/home/yangming/anaconda3/envs/pai/bin/python infer_att3.py \
  --ckpt checkpoints/ours_selected_best.pt --device cuda:0
```

## 说明

官方仓库的部分代码基于[kangverse/EASE](https://github.com/kangverse/EASE) [LNLN](https://github.com/Haoyu-ha/LNLN)。本目录只借用其工程布局（`configs` / `core` / `models` / `train.sh` / `robust_eval.sh`），算法与损失以本目录实现为准。

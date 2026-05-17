# HeatRAD — 项目上下文（用于新会话快速恢复）

> 工程视角速查。详细周报见 `reports/weekly_*.md`。

---

## 1. 是什么
3D 胸部 CT → 结构化放射学报告的端到端模型。架构（见 `heatrad/models/`）：

```
vol (B,3,D,H,W)
  └─ HeatConductionSliceEncoder   (vHeat-Tiny 改造 + 1-D DCT 切片轴)
        └─ AnomalyHead             (α_j 切片可疑度)
        └─ OrganCrossAttention     (K=32 器官文本锚作 query → q̃_{k,j})
              └─ ReportDecoder     (sparsemax 选片 → 视觉前缀 → Qwen3-8B+LoRA)
                    └─ generate / autoregressive L_gen
```

* 训练目标：仅 `L_gen` 自回归报告损失。
* 可训练参数 ≈ 227 M（视觉全可训 + 解码器仅 LoRA r=64）。

---

## 2. 关键路径

| 用途 | 路径 |
|---|---|
| 项目根 | `/data/DaZhou_data/heatrad` |
| 数据 | `/data/DaZhou_data/RadGenome-ChestCT` |
| 预处理 cache | `cache/vol_npz/` (25693 个 .npz, shape `(3,64,224,224)` z-score)、`cache/report_tok/`、`cache/manifest.parquet`（路径已 rewrite 到当前根） |
| 模型权重 | `checkpoints/vheat/vheat_tiny_in1k.pth`、`checkpoints/qwen3/Qwen3-8B/`、`checkpoints/biomedbert/` |
| 训练日志 + ckpt | `runs/heatrad_h20/` |
| 训练完成的最新 ckpt | `runs/heatrad_h20/ckpt/last.pt`（= step_044000, 17.3 GB） |
| 周报 | `reports/weekly_2026-05-11.md` |

---

## 3. 环境

```bash
# conda env (Python 3.10, PyTorch 2.5.1+cu124, transformers 4.55.0)
source /home/zhouly/miniconda3/etc/profile.d/conda.sh
conda activate heatrad
source scripts/env.sh        # 设置 PATH、CUDA_VISIBLE_DEVICES、NCCL 三连
```

`scripts/env.sh` 关键变量：
```bash
HEATRAD_ROOT=/data/DaZhou_data/heatrad
RADGENOME_PATH=/data/DaZhou_data/RadGenome-ChestCT
CUDA_VISIBLE_DEVICES=2,3,4,5         # 本机 GPU 0/1 长期被他人占用
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_NVLS_ENABLE=0                   # H20-3e split-mode 必须
```

未装 nvcc → flash-attn 装不上 → decoder 走 SDPA。

---

## 4. 训练

```bash
bash scripts/run_train.sh                                    # 从零开始
# 或恢复：
bash scripts/run_train.sh "+resume_from=runs/heatrad_h20/ckpt/last.pt"
```

关键配置（合成自 hydra）：

| 项 | 值 | 文件 |
|---|---|---|
| Effective batch | 16 (bsz/gpu=1 × 4 GPU × grad_accum=4) | `configs/profile/h20_4gpu.yaml` |
| Optimizer | AdamW(0.9, 0.999), wd=0.05 | `configs/train/main.yaml` |
| LR vis / lora (peak) | 1e-4 / 1e-4 | 同上 |
| Schedule | 1485 step warmup + cosine | 同上 |
| Total steps | 45000 (≈30 epoch) | 同上 |
| Abstain init | -1.0 | `configs/model/heatrad_qwen3_8b.yaml` |
| Organ-bag dropout | 0.5 → 0 前 50% 退火 | 训练代码硬编码 |
| Precision | bf16 autocast + grad-checkpoint(decoder) | profile |

---

## 5. 推理 / 评测

```bash
CUDA_VISIBLE_DEVICES=2 python -m heatrad.eval.run_eval \
  "+ckpt=$HEATRAD_ROOT/runs/heatrad_h20/ckpt/last.pt" \
  "+eval.n_samples=50" \
  "+eval.out=$HEATRAD_ROOT/runs/heatrad_h20/eval/val_50.json"
```

会算 3 类指标：
* **NLG**：BLEU-1, BLEU-4, METEOR, ROUGE-L
* **CE**：micro-averaged entity Precision / Recall / F1（lexicon 由 `train_case_disorders.csv` 构造）
* **Clinical**：RaTEScore（MAGIC-AI4Med）

---

## 6. 工程上踩过的坑

| 坑 | 修复 |
|---|---|
| vHeat 的 `to_k` 用 ReLU 强制频率门非负，导致 signed gate 不生效 | 删除 ReLU（`heatrad/external/hco/vheat.py`），并对 `pow(weight_exp, k_signed)` 的指数 clamp 到 [-2, 4] 防 overflow |
| Heat1D 用 rFFT proxy 不等价 DCT | 改成 cosine 基 einsum 实现真 DCT-II |
| 初版 `init_abstain=+1.0` 让 sparsemax 全部 abstain，视觉 token 梯度归零 | 改 `init_abstain=-1.0`，并把 `abstain / log_tau / log_tau_o` 排除出 weight decay |
| LoRA LR 2e-4 在 cosine 峰值时 loss 突冲到 5-7 + α 坍塌 | 减半到 1e-4 后稳定 |
| Qwen3 在 transformers 4.46 不支持 | 升级到 4.55（torch 2.5.1 兼容） |
| BiomedBERT safetensors 缺 metadata，transformers 加载报 None.get | 用默认加载策略，4.55 已修复 |
| H20-3e split-mode 不支持 NVLink-Sharp | NCCL_NVLS_ENABLE=0 |
| GPUs 0/1 被他人长期占用 | CUDA_VISIBLE_DEVICES=2,3,4,5 |

---

## 7. 当前状态

* 训练已完成（step 45000/45000，2026-05-12 ~11:00）。
* 最终 loss ≈ 0.03–0.04，α[max/med] ≈ 0.221/0.218 稳定，abstain=0。
* 22 个 checkpoint 落盘，`last.pt` = step_044000（step_045000 写盘时被进程退出的 NCCL 噪音打断，权重差异可忽略）。
* 推理 / val 评测脚本已写：`heatrad/eval/run_eval.py`、`heatrad/eval/metrics.py`。

---

## 8. 下一步可能要做

* val 1564 个样本全量评测（≈ 2 h on 1×H20）
* 把 `eval/run_eval.py` 升级为多 GPU 并行推理
* 实现 visualization：α 沿切片轴的曲线 + 每器官 sparsemax 选片图
* 接入 RaTEScore 之外的 GREEN（需下载额外模型）
* Demo 包装（Gradio / 截图）

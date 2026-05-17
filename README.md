# HeatRAD

3D 胸部 CT → 结构化放射学报告的端到端模型。
架构:`HeatConductionSliceEncoder`(vHeat-Tiny + 1-D DCT 切片轴)→ `AnomalyHead`(切片可疑度 α)→ `OrganCrossAttention`(K=32 器官文本锚)→ `ReportDecoder`(sparsemax 选片 → Qwen3-8B+LoRA)。

> 详细工程上下文见 `CONTEXT.md`,周报见 `reports/weekly_*.md`。

---

## 仓库分支

| 分支 | 用途 | 数据 | 报告语言 |
|---|---|---|---|
| `main` | 英文基线 pipeline(已完整训练 + 评测) | RadGenome-ChestCT | 英文 |
| `feat/zh-pipeline` | 中文 pipeline:MAE 自监督预训练编码器 → 达州数据中文端到端后训练 | RadGenome(SSL)+ 达州 CT(E2E) | 中文 |

切换:`git checkout main` / `git checkout feat/zh-pipeline`

---

## 环境

```bash
source /home/zhouly/miniconda3/etc/profile.d/conda.sh
conda activate heatrad                 # Python 3.10, torch 2.5.1+cu124, transformers 4.55.0
cd /data/DaZhou_data/heatrad
source scripts/env.sh                  # HEATRAD_ROOT / RADGENOME_PATH / CUDA_VISIBLE_DEVICES=2,3,4,5 / NCCL 三连
```

关键路径:

| 用途 | 路径 |
|---|---|
| 项目根 | `/data/DaZhou_data/heatrad` |
| RadGenome 数据 | `/data/DaZhou_data/RadGenome-ChestCT` |
| 达州 CT 原始 DICOM | `/data/DaZhou_data/达州CT数据/临床科研图像数据`(16407 病人) |
| 达州报告 Excel | `/data/DaZhou_data/达州CT数据/临床研究中心放射检查信息.xlsx`(中文,30000 行) |
| RadGenome 体积 cache | `cache/vol_npz/*.npz`(25693 个,`(3,64,224,224)`) |
| 训练日志 + ckpt | `runs/<run_name>/` |

---

## `main` 分支:英文基线

### 训练
```bash
bash scripts/run_train.sh
# 恢复:bash scripts/run_train.sh "+resume_from=runs/heatrad_h20/ckpt/last.pt"
```
最新 ckpt:`runs/heatrad_h20/ckpt/last.pt`(step 44000)。

### 评测(全量 val 1564)
```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python -m heatrad.eval.run_eval \
  +ckpt=runs/heatrad_h20/ckpt/last.pt +eval.n_samples=0 \
  +eval.out=runs/heatrad_h20/eval/val_full.json
# 仅重算指标(已有 *_raw.json,不重新推理):
CUDA_VISIBLE_DEVICES=2 python -m heatrad.eval.rescore \
  runs/heatrad_h20/eval/val_full_raw.json runs/heatrad_h20/eval/val_full.json
```
指标:NLG(BLEU/METEOR/ROUGE)、CE(自建词典实体 F1)、RadGraph F1、CheXbert F1、RaTEScore。
最近结果:`runs/heatrad_h20/eval/val_full.json`(BLEU-1 0.443 / RaTEScore 0.625)。

---

## `feat/zh-pipeline` 分支:中文 pipeline

四个阶段,顺序执行,每步独立可验证。

### 阶段 1 — 统一预处理(对齐 RadGenome ↔ 达州)
> 待实现:`heatrad/data/ct_preprocess.py`(DICOM→真HU→统一开窗→224→D=64→ImageNet z-score,与 RadGenome cache 字节兼容)、`heatrad/data/build_dazhou_manifest.py`(Excel 清洗 + AccessionNumber 对齐 + 严格胸部CT平扫过滤,产出 `cache_zh/manifest_zh.parquet`)。

### 阶段 2 — heat-encoder MAE 自监督预训练(RadGenome)✅ 已实现
SimMIM 式掩码重建,只训编码器 + 轻量重建头,复用 RadGenome cache(不需报告)。

```bash
# 冒烟测试(1 GPU, 6 步)
CUDA_VISIBLE_DEVICES=2 python -m heatrad.train.pretrain_mae \
  train=mae_pretrain run_name=zh_mae_smoke \
  ++train.total_steps=6 ++train.save_every=6 ++train.bsz_per_gpu=1 ++train.grad_accum=1

# 正式训练(4 GPU)
source scripts/env.sh
torchrun --nproc_per_node=4 -m heatrad.train.pretrain_mae \
  train=mae_pretrain run_name=zh_mae
```
- 配置:`configs/train/mae_pretrain.yaml`(mask_ratio 0.6, patch 32, grid 7, 30000 步, lr 1.5e-4)
- 代码:`heatrad/models/mae_head.py`、`heatrad/train/pretrain_mae.py`
- 产出:`runs/zh_mae/ckpt/encoder_last.pt`(只含编码器权重,供阶段 3 热启)
- 日志:`/tmp/zh_mae.log`

### 阶段 3 — 中文端到端后训练(达州)
> 待实现:编码器用 `runs/zh_mae/ckpt/encoder_last.pt` 热启,中文 prompt 模板,训练时 mask 掉 prompt 前缀 token 的 loss,`configs/data/dazhou.yaml` + `configs/train/zh_e2e.yaml`。

### 阶段 4 — 中文评测
> 待实现:中文分词(jieba)BLEU/ROUGE + 自建中文实体词典 CE。

---

## 重启速查(对话丢失后)

1. `cd /data/DaZhou_data/heatrad && git branch`(确认在哪个分支)
2. `git log --oneline -5`(看最近进度)
3. 看 `CONTEXT.md` + 最新 `reports/weekly_*.md`
4. MAE 训练进度:`tail -50 /tmp/zh_mae.log`;ckpt:`ls runs/zh_mae/ckpt/`
5. 继续/恢复:按上面对应阶段的命令重跑(MAE 当前不支持 resume,从头或改 total_steps)

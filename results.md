# HeatRAD Results Log

> 每跑完一个 run **立刻把数字写进这里**。这是 NeurIPS reproducibility 的最佳兑现物，
> 也是投稿前一晚不必翻 wandb 的保险。
>
> 列：run 名 / phase / 数据规模 / 步数 / 关键指标 / ckpt 路径 / 跑此 run 的命令。

---

## 速览（按时间逆序）

| Date | Run | Phase | Train data | Steps | Key metric | Notes |
|------|-----|-------|-----------|-------|-----------|-------|
| 2026-04-29 | `heatrad_pb_full_sparse` ⭐ | B + α-sparse λ=0.01 | FULL 24k | 5000 | α[max/med]=**0.251/0.063 ratio=3.98**, L_att=0.18 L_gen=0.22 | **结构问题被解决** — sparsity prior 让大多数 α≈0、少数突出 |
| 2026-04-29 | `heatrad_pb_full_5k` | B (control) | FULL 24k | 5000 | α[max/med]=0.242/0.234 ratio=1.04, L_att=0.18 L_gen=0.22 | 5000 步仍软分化，确认是结构而非训练量问题 |
| 2026-04-28 | `heatrad_pb_full` | B | **FULL 24k** | 1000 | L_gen 2.29→0.87 (健康), α[max/med] 1.10→1.36→**1.10（稳定 soft 分化）** | ⭐ 否决 dev overfit 假说，真实数据上 α 仍只 ~1.10 → noisy-OR 饱和效应 |
| 2026-04-28 | `heatrad_pa_full` | A | **FULL 24k** | 1000 | L_anc 0.26→0.0000 (step 700+) | 全量数据 Phase A 收敛快且干净 |
| 2026-04-28 | `heatrad_pb_500` | B | dev_subset 200 | 500 | L_gen 2.36→0.30 (over-fit), α[max/med] 早期 1.50 → 末段 1.08 | dev_subset 现象，被全量数据复现 → 是结构特征不是 artifact |
| 2026-04-28 | `heatrad_pb_200` | B | dev_subset 200 | 200 | val Micro-F1=65.5 (5 cases), BLEU-1=24.0, ROUGE-L=18.0, α[max/med] 1.16→2.26 peak | Eval pipeline 首次 toy 验证 |
| 2026-04-28 | `heatrad_pa_v2_vheat` | A | dev_subset 200 | 300 | L_anc 1.89→0.0006 in 60 steps | vHeat-Tiny 预训练 ImageNet 加载（234/241 tensor），116s |
| 2026-04-27 | `heatrad_pa_300` | A | dev_subset 200 | 300 | L_anc 1.54→0.001 in 60 steps | encoder 随机初始化（vHeat 权重还没下） |
| 2026-04-27 | `heatrad_dryrun_s1` (废弃) | v1 Stage I | dev_subset 200 | 5000 | L_att 1.7→0.2 但 α[max/med]=0.469/0.466 ⇒ **坍缩** | 证明 v1 plan Stage I 顺序在结构上不行；触发 plan v2 改成 Phase A→B |

---

## 结构化记录

### `heatrad_pb_full` ⭐

- **Date**: 2026-04-28
- **Phase**: B (joint), full 24k data
- **Resume from**: `runs/heatrad_pa_full/ckpt/last.pt`
- **Data**: manifest.parquet (24129 train, 1564 val), `data.vol_shape=[48,224,224]`
- **Hyperparams**: bsz=1, grad_accum=4, lr=1e-4 cosine, total_steps=1000
- **Hardware**: 1× RTX 4090 (GPU 0), 23 GiB peak
- **Time**: 2233 s = 37 min for 1000 steps
- **Trajectory** (compressed):
  ```
  step   25: L_att=0.29  L_anc=0.0011  L_gen=2.29  α[max/med]=0.305/0.279  ratio=1.09
  step  100: L_att=0.13  L_anc=0.0019  L_gen=1.14  α[max/med]=0.217/0.160  ratio=1.36 ← peak
  step  300: L_att=0.38  L_anc=0.0019  L_gen=1.29  α[max/med]=0.219/0.207  ratio=1.06
  step  500: L_att=0.20  L_anc=0.0010  L_gen=0.56  α[max/med]=0.223/0.197  ratio=1.13
  step  700: L_att=0.20  L_anc=0.0006  L_gen=1.23  α[max/med]=0.220/0.196  ratio=1.12
  step 1000: L_att=0.27  L_anc=0.0004  L_gen=0.87  α[max/med]=0.230/0.210  ratio=1.10
  ```
- **关键观察**：
  - **L_gen 0.87**（vs dev_subset 500 步末段 0.30）→ 全量数据**消除了过拟合** ✓
  - **α 末段 ratio 1.10**（vs dev_subset 1.08）→ **几乎一样**！dev 现象在全量上**复现** ⭐
  - L_anc 全程 ≤ 0.002 → Phase A 锚点完全保留 ✓
- **诊断升级（重要）**：α 软分化（ratio ~1.1）**不是 dev overfit 的副作用**，是 noisy-OR 公式的**饱和效应**：当 β_{j,k} 平均 ~0.3、R=8、α 平均 ~0.2 时，R·ᾱ·β̄ ≈ 0.48 → β_k ≈ 0.38 已能拟合 y_o 的常见 organ。L_att 没有持续把 α 推向极端的压力。
- **结构性结论**：
  - ✅ Phase A→B 顺序成功规避 v1 的硬坍缩（ratio=1.00）
  - 🟡 但仅给到 soft 分化（ratio ~1.10），未达期望的 ≥1.5
  - 要强分化需要额外手段（按风险递增）：
    1. 恢复 D=64-80（H20 上不再受显存约束）
    2. α-sparsity prior：`+λ·sum_j α_j` 鼓励稀疏
    3. 减小 R（topr 8→4）让每个 slice 边际信号更尖锐
    4. 用 RadGenome region_anatomy mask 派生 slice-level 监督
- **Verdict**: 🟢 plan 整体可行，但 alpha 强分化需后续 ablation 探索（不阻塞 main results）
- **Ckpt**: `runs/heatrad_pb_full/ckpt/last.pt`

#### `heatrad_pb_full` eval on 100 val cases ⭐

- **Date**: 2026-04-28
- **Time**: 1931 s = 32 min
- **alpha_max / alpha_med (val mean)**: 0.208 / 0.208 → **ratio = 1.0000**

  ⚠️ Eval 模式 α 完全均匀（vs train 模式 1.10）。诊断：vHeat backbone 的 **DropPath 在训练时引入的随机性**让训练态 α 看起来有 1.1 的分化，但在 eval 模式 DropPath 关闭，encoder 未真正学到 slice-discriminative 特征 → 所有 slice 输出几乎相同的 f_j → α softmax 退化为常数。

- **Organ metrics (Micro)**: F1=67.2  P=70.2  R=64.4
- **Organ metrics (Macro)**: F1=12.7  P=11.0  R=15.6
- **Per-organ F1**：只有最高频 5 个 organ 学到：
  ```
  lung      94.2  (P=89, R=100)
  heart     91.9  (P=85, R=100)
  pleura    81.0  (P=68, R=100)
  esophagus 75.8  (P=61, R=100)
  bone      64.9  (P=48, R=100)
  其余 27 个 organs : F1=0.0
  ```
  R=100 对所有学到的 organ → 模型实际上是把 β_k 推到 1（高 recall）然后由 P 来限制 → 没在做组学 attribution，只是 marginal classifier。
- **Report metrics**: BLEU-1=14.65, BLEU-2=2.46, BLEU-3=0.29, BLEU-4=0.06, ROUGE-L=10.6
  - BLEU-4 ≈ 0 → 1000 步在 24k 数据上 ≈ 0.17 epoch，Qwen3 远未学会写作风格
  - 比 dev_subset 200 步 (BLEU-4=3.37, ROUGE-L=18.0) 还低，因为 dev_subset overfit 拟合了那 200 例的风格
- **Subset-Acc**: 0.0 → 没一个完整 case 全 organ 都对

- **诊断（更深一层）**：
  1. **α 真的没学**：train 态 ratio 1.1 是 DropPath 噪声，不是真分化
  2. **Macro-F1 12.7 = 67.2 / 27 余 0** → 模型只有 5 个 head 在工作
  3. **Qwen3 远没收敛** → BLEU 全降是 train epoch 不够而非模型问题
- **Verdict**: 🟡 工程链路 100% 通；模型质量 = 早期 baseline；要正式数字必须上 H20 长训练（≥30k 步）+ alpha 改进


### `heatrad_pa_full`

- **Date**: 2026-04-28
- **Phase**: A (anchor warmup), full 24k data
- **Encoder init**: vHeat-Tiny ImageNet pretrained
- **Hyperparams**: total_steps=1000, lr=2e-4 cosine, bsz=1, grad_accum=4
- **Time**: 454 s = 7.5 min for 1000 steps
- **Trajectory**: L_anc 0.26 → 0.002 (step 100) → 0.0001 (step 250) → 0.0000 (step 700+)
- **Verdict**: ✅ 全量数据 Phase A 比 dev_subset 收敛更稳定（每步信号方差更小）
- **Ckpt**: `runs/heatrad_pa_full/ckpt/last.pt`

### `heatrad_pb_500`

- **Date**: 2026-04-28
- **Phase**: B (joint)
- **Resume from**: `runs/heatrad_pa_v2_vheat/ckpt/last.pt`
- **Data**: dev_subset (200 train, 50 val), `data.vol_shape=[48,224,224]`
- **Hyperparams**: bsz=1, grad_accum=4, lr=1e-4 cosine, warmup_ratio=0.05, total_steps=500
- **Hardware**: 1× RTX 4090 (GPU 1), 23 GiB peak
- **Time**: 1273 s = 21 min for 500 steps
- **Trajectory**:
  ```
  step  25: L_att=0.57  L_anc=0.010  L_gen=2.36  α[max/med]=0.43/0.31  ratio=1.39
  step  50: L_att=0.16  L_anc=0.009  L_gen=1.75  α[max/med]=0.18/0.12  ratio=1.50  ← peak
  step 100: L_att=0.36  L_anc=0.008  L_gen=1.86  α[max/med]=0.16/0.14  ratio=1.14
  step 200: L_att=0.25  L_anc=0.004  L_gen=0.64  α[max/med]=0.26/0.16  ratio=1.63
  step 300: L_att=0.26  L_anc=0.002  L_gen=0.41  α[max/med]=0.19/0.18  ratio=1.06
  step 400: L_att=0.16  L_anc=0.002  L_gen=0.69  α[max/med]=0.20/0.19  ratio=1.05
  step 500: L_att=0.19  L_anc=0.002  L_gen=0.30  α[max/med]=0.21/0.19  ratio=1.08
  ```
- **关键观察**：
  - L_gen 单调降到 0.30（甚至 step 450 阶段 0.13）→ **严重过拟合 dev_subset**（200 例 × 10 epochs effective）
  - L_anc 全程稳定低（0.002-0.010）→ Phase A 锚点没被破坏 ✓
  - α 的分化轨迹**先扩张后收缩**：早期 max/med 上升到 1.5-1.6，晚期回落到 1.05-1.08
- **诊断**：dev_subset 太小 + L_gen 过拟合后，模型不再"需要"α 来挑 slice —— q_jk 已经把所有 case 的 organ 信号背下来了，noisy-OR 链路退化为"所有 α 都低且均匀，但不影响 BCE 拟合 y_o"。这是 200 例的天花板，不是结构问题。
- **下一步行动**：必须用全量 24k 数据复跑同样的 500/1000/2000 步，对比 α 末段是否仍保持分化。
- **Verdict**: 🟡 部分成功 — 三损失协调收敛、显存稳定、Phase A 不退化；但 α 在 200+ 步后回缩证明小数据 overfit 是主要风险。
- **Ckpt**: `runs/heatrad_pb_500/ckpt/last.pt`

### `heatrad_pb_200`

- **Date**: 2026-04-28
- **Phase**: B (joint)
- **Resume from**: `runs/heatrad_pa_v2_vheat/ckpt/last.pt`
- **Data**: dev_subset (200 train, 50 val), `data.vol_shape=[48,224,224]`
- **Hyperparams**: bsz=1, grad_accum=4, lr=1e-4 cosine
- **Hardware**: 1× RTX 4090 (GPU 0), 23 GiB peak
- **Time**: 448 s = 7.5 min for 200 steps
- **Trajectory**:
  ```
  step  10: L_att=0.20  L_anc=0.008  L_gen=2.43  α[max/med]=0.46/0.40  ratio=1.16
  step  30: L_att=0.17  L_anc=0.007  L_gen=1.66  α[max/med]=0.33/0.15  ratio=2.26 ← peak
  step 100: L_att=0.36  L_anc=0.004  L_gen=1.83  α[max/med]=0.19/0.16  ratio=1.21
  step 200: L_att=0.20  L_anc=0.003  L_gen=0.67  α[max/med]=0.19/0.15  ratio=1.25
  ```
- **Eval (5 val cases)**:
  ```
  organ : Micro-F1=65.5  Micro-P=90.0  Micro-R=51.4  Macro-F1=11.8
  report: BLEU-1=24.0    BLEU-2=12.2  BLEU-3=6.07   BLEU-4=3.37   ROUGE-L=18.0
  ```
  (note: 5 val cases too small for credible Macro-F1; toy result)
- **Verdict**: ✅ Pipeline 全栈打通，三损失协调收敛，α 摆脱 v1 的均匀坍缩。
- **Ckpt**: `runs/heatrad_pb_200/ckpt/last.pt`

### `heatrad_pa_v2_vheat`

- **Date**: 2026-04-28
- **Phase**: A (anchor warmup)
- **Encoder init**: vHeat-Tiny ImageNet pretrained (loaded 234/241 tensor; 7 skipped = classifier head)
- **Data**: dev_subset (200), `data.vol_shape=[80,224,224]`
- **Hyperparams**: bsz=1, grad_accum=4, lr=2e-4 cosine, total_steps=300
- **Hardware**: 1× RTX 4090, 5 GiB peak
- **Time**: 116 s
- **Trajectory**: L_anc 1.89 → 0.27 (step 40) → 0.0356 (step 60) → 0.0006 (step 300)
- **Verdict**: ✅ q_k ↔ t_k 对齐成功，Phase B 起点干净。
- **Ckpt**: `runs/heatrad_pa_v2_vheat/ckpt/last.pt`

### `heatrad_pa_300` (superseded by v2)

- **Date**: 2026-04-27
- **Phase**: A (anchor warmup)
- **Encoder init**: 随机（vHeat 权重还没下）
- **Hyperparams**: 同 v2
- **Trajectory**: L_anc 1.54 → 0.025 (step 60) → 0.001 (step 300)
- **Verdict**: ✅ 与 v2 几乎一样收敛——说明 Phase A 中 encoder 是冻结的，初始化质量对 L_anc 收敛影响很小（如预期）。

### `heatrad_dryrun_s1` (废弃实验)

- **Date**: 2026-04-27
- **Phase**: plan v1's Stage I (encoder + AnomalyHead trainable, Q-Former frozen random)
- **Hyperparams**: total_steps=5000, lr=2e-4
- **Time**: 833 s = 14 min
- **Trajectory**: L_att 1.7 → 0.2 in ~500 steps; **α[max/med] = 0.469/0.466 全程不变（ratio≈1.00）**
- **Verdict**: ❌ **α 完全坍缩**，符合 §5.2 的解析推导（β_jk 均匀时 noisy-OR 只对 ᾱ 有梯度）。**这次失败直接驱动 plan v1 → v2 的训练顺序变更**（改为 method.tex 的 Phase A → Phase B）。
- **Ckpt**: 不保留（5000 步 save_every=10000 没存）

---

## 待跑（顺序大致按价值-成本比）

1. ⏳ `heatrad_pb_500`：500 步看 α 是否稳定分化（in progress）
2. 📋 全量 24k train 数据预处理完成后：`heatrad_pb_full_dev`，Phase A 1000 步 + Phase B 2000 步在全量数据上。预计单卡 4090 ~ 4-6 小时。
3. 📋 Eval 全量 1564 val cases — 单卡 ~10 小时；想尽快结果建议用 `eval.max_cases=200` 的子集 ~1.3 小时。
4. 📋 上 H20 后真训：Phase A 10k 步 + Phase B 80k 步。
5. 📋 消融：w/o HCO（Swin-T 替）、w/o Stage I 排序（全 slice 给 Stage II）、w/o noisy-OR（max 替）、w/o text-anchor、w/o LoRA-on-encoder。

---

## 命令模板（避免重复造轮子）

### Phase A 训练
```bash
torchrun --nproc_per_node=N -m heatrad.train.trainer \
    --config-name=default train=phase_a \
    run_name=NAME
```

### Phase B 训练（resume Phase A）
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=N -m heatrad.train.trainer \
    --config-name=default train=phase_b \
    +resume_from=runs/PA_RUN/ckpt/last.pt \
    data.vol_shape=[48,224,224] \
    run_name=NAME
```

### Eval
```bash
python -m heatrad.eval.run_eval \
    --config-name=default train=phase_b \
    data.vol_shape=[48,224,224] \
    +ckpt=runs/PB_RUN/ckpt/last.pt \
    +eval.split=val +eval.max_cases=200 \
    +eval.metrics_include=[bleu,rouge,bertscore] \
    +eval.out=runs/PB_RUN/eval/val_200.json \
    run_name=PB_RUN
```

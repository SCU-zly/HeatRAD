# HeatRAD Third-Party Asset Manifest

This file tracks all vendored code and pretrained weights, with version,
source, and license. Maintained per the NeurIPS 2025 reproducibility
checklist (item 12).

## Vendored code

| Local path | Source | License |
|------------|--------|---------|
| `heatrad/external/hco/vheat.py` | `vHeat/classification/models/vHeat.py` (https://github.com/MzeroMiko/vHeat) | upstream repo, see https://github.com/MzeroMiko/vHeat (no top-level LICENSE file at the time of vendoring; treat as research-only until clarified) |
| `heatrad/external/anomaly_ov_ref/llava_model/` | `Anomaly-OneVision/llava/model/` (selected files: `anomaly_expert.py`, `llava_arch.py`, `language_model/llava_qwen.py`) | LICENSES/anomaly_ov.LICENSE |

## Pretrained weights

| Local path | Model | Source | License |
|------------|-------|--------|---------|
| `checkpoints/vheat/vheat_tiny_in1k.pth` | vHeat-Tiny ImageNet-1k | vHeat official release | research-only (per upstream README) |
| `checkpoints/qwen3/Qwen3-8B-Instruct/` | Qwen3-8B-Instruct | HuggingFace `Qwen/Qwen3-8B-Instruct` | Tongyi Qianwen LICENSE |
| `checkpoints/biomedbert/` | BiomedNLP-PubMedBERT-base | HuggingFace `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | MIT |
| `checkpoints/blip2_qformer/` | BLIP-2 Q-Former (init) | HuggingFace `Salesforce/blip2-flan-t5-xl` (qformer subset) | BSD-3-Clause |

"""Forward-only smoke test: encoder + stage1 + stage2 on random tensors.

Skips text_anchor (needs BiomedBERT weights) and decoder (needs Qwen3 weights).
Run on a single GPU:
    python scripts/smoke_test_forward.py
"""
import torch
from heatrad.models.encoder import EncoderConfig, HeatConductionSliceEncoder
from heatrad.models.stage1_anomaly import Stage1Config, AnomalyHead
from heatrad.models.stage2_qformer import (
    Stage2Config, OrganQFormer, noisy_or, slice_aggregated_q, significance_map,
)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Tiny dimensions: D=8, H=W=128, K=8 organs, B=1.
    B, C_in, D, H, W = 1, 3, 8, 128, 128
    K = 8

    enc_cfg = EncoderConfig(
        img_size=H, patch_size=4, in_chans=C_in,
        depths=(1, 1, 2, 1), dims=(48, 96, 192, 384),
        gradient_checkpointing=False, max_d=32,
    )
    encoder = HeatConductionSliceEncoder(enc_cfg).to(device, dtype)

    s1_cfg = Stage1Config(c_l=enc_cfg.dims[-1], topr=4)
    head1 = AnomalyHead(s1_cfg).to(device, dtype)

    s2_cfg = Stage2Config(c_l=enc_cfg.dims[-1], d=384, k=K, n_layers=2, n_heads=8)
    qf = OrganQFormer(s2_cfg).to(device, dtype)

    vol = torch.randn(B, C_in, D, H, W, device=device, dtype=dtype)
    feats = encoder(vol)
    print("encoder feats:", tuple(feats.shape))  # expect (B, D, N_L, C_L)

    alpha, topk_idx, _ = head1(feats)
    print("alpha:", tuple(alpha.shape), "topk_idx:", tuple(topk_idx.shape))
    alpha_top = torch.gather(alpha, 1, topk_idx)
    print("alpha_top:", tuple(alpha_top.shape))

    q_jk, A_jk, beta_jk = qf(feats, topk_idx)
    print("q_jk:", tuple(q_jk.shape), "A_jk:", tuple(A_jk.shape), "beta_jk:", tuple(beta_jk.shape))

    beta_k = noisy_or(alpha_top, beta_jk)
    qbar_k = slice_aggregated_q(alpha_top, q_jk)
    M = significance_map(beta_jk, A_jk)
    print("beta_k:", tuple(beta_k.shape), "qbar_k:", tuple(qbar_k.shape), "M:", tuple(M.shape))

    # Tiny BCE backward to confirm grads flow.
    y_o = torch.randint(0, 2, (B, K), device=device).float()
    loss = torch.nn.functional.binary_cross_entropy(beta_k.clamp(1e-6, 1 - 1e-6), y_o)
    loss.backward()
    print(f"loss = {loss.item():.4f}, grads OK")

    # Verify a few key params have non-None gradients.
    n_with_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
    n_total = sum(1 for _ in encoder.parameters())
    print(f"encoder params with grad: {n_with_grad}/{n_total}")
    print(f"qf.queries grad: {'OK' if qf.queries.grad is not None else 'NONE'}")


if __name__ == "__main__":
    main()

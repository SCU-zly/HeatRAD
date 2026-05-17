"""Sanity-check the heatrad conda env and GPUs."""
import torch

print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  cap={p.major}.{p.minor}  mem={p.total_memory/2**30:.1f} GiB")

assert torch.cuda.is_available(), "CUDA not available"

# dev_4090 expectation: 4x RTX 4090 (sm_89, ~24 GiB)
n = torch.cuda.device_count()
if n != 4:
    print(f"[warn] expected 4 GPUs (dev_4090 profile), got {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    if (p.major, p.minor) != (8, 9):
        print(f"[warn] device {i} cap={p.major}.{p.minor}, expected 8.9 (Ada)")

print("nccl available:", torch.distributed.is_nccl_available())
print("flash-attn available:", end=" ")
try:
    import flash_attn  # noqa: F401
    print(flash_attn.__version__)
except ImportError as e:
    print(f"NO ({e})")

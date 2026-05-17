"""Tiny DDP setup helpers."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_ddp() -> tuple[int, int, int]:
    """Initialize DDP from env (torchrun); return (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, ws, local


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def ddp_barrier() -> None:
    if is_dist():
        dist.barrier()

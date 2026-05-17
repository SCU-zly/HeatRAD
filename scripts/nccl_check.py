"""Multi-GPU NCCL allreduce sanity check.

Run with:
    torchrun --standalone --nproc_per_node=4 scripts/nccl_check.py
"""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    x = torch.ones(1 << 26, device="cuda") * rank  # 256 MiB f32
    dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        expected = sum(range(world_size))
        print(f"allreduce ok, sum = {x[0].item()} expected = {expected}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

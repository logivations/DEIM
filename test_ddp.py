import torch, os

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("RANK", os.environ.get("RANK"))
print("LOCAL_RANK", os.environ.get("LOCAL_RANK"))
print("WORLD_SIZE", os.environ.get("WORLD_SIZE"))

try:
    import torch.distributed as dist
    print("torch.distributed is available.")
    print("Backends available:", dist.is_gloo_available(), dist.is_nccl_available(), dist.is_mpi_available())
except ImportError as e:
    print("torch.distributed is NOT available:", e)


if dist.is_available() and not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")

if dist.is_initialized():
    rank = dist.get_rank()
    print(f"Hello from rank {rank}")
else:
    print("Distributed not initialized")


"""Run a small CUDA matrix workload when PyTorch is installed."""
import sys
try:
    import torch
except ImportError:
    print("PyTorch is not installed. Install a CUDA-enabled PyTorch build first.")
    sys.exit(2)

if not torch.cuda.is_available():
    print("CUDA is unavailable. Check the NVIDIA driver and container runtime.")
    sys.exit(1)

device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)
print(f"GPU: {props.name}")
print(f"Compute capability: {props.major}.{props.minor}")
a = torch.randn((2048, 2048), device=device)
b = torch.randn((2048, 2048), device=device)
c = a @ b
torch.cuda.synchronize()
print(f"CUDA smoke test passed; result mean={c.mean().item():.6f}")

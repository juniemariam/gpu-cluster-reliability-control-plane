"""Small real CUDA workload used by the platform's execute endpoint."""
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        print("torch is not installed; install requirements-gpu.txt", flush=True)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        print("CUDA is unavailable", flush=True)
        raise SystemExit(3)
    device = torch.device("cuda")
    print(f"running CUDA workload on {torch.cuda.get_device_name(0)}", flush=True)
    end = time.time() + args.seconds
    while time.time() < end:
        a = torch.randn((2048, 2048), device=device)
        b = torch.randn((2048, 2048), device=device)
        _ = a @ b
        torch.cuda.synchronize()
    print("CUDA workload completed", flush=True)

if __name__ == "__main__": main()

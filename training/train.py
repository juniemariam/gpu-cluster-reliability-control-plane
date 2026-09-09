"""Real synthetic image-classification workload for a Kubernetes GPU Job."""
import argparse, time
import torch
from torch import nn

def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=100); p.add_argument("--batch-size", type=int, default=256); args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this training job")
    device = torch.device("cuda")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
    print(f"training on {torch.cuda.get_device_name(0)}", flush=True)
    started = time.time()
    for step in range(args.steps):
        inputs = torch.randn(args.batch_size, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (args.batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(inputs), labels); loss.backward(); optimizer.step()
        if step % 10 == 0: print(f"step={step} loss={loss.item():.4f}", flush=True)
    torch.cuda.synchronize(); print(f"completed steps={args.steps} elapsed={time.time()-started:.2f}s", flush=True)

if __name__ == "__main__": main()

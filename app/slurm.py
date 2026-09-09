"""Optional Slurm adapter; safely reports unavailable when running locally."""
import shutil, subprocess

class SlurmScheduler:
    def available(self): return all(shutil.which(x) for x in ("sbatch", "squeue", "scancel"))
    def submit(self, command, gpus=1, partition="gpu"):
        if not self.available(): return {"backend": "slurm", "status": "unavailable", "message": "sbatch/squeue/scancel not found; use Kubernetes or local scheduler"}
        script = "#!/bin/bash\n" + " ".join(command)
        result = subprocess.run(["sbatch", f"--gpus={gpus}", f"--partition={partition}"], input=script, text=True, capture_output=True, check=False)
        return {"backend": "slurm", "status": "submitted" if result.returncode == 0 else "failed", "output": result.stdout or result.stderr}
    def cancel(self, job_id):
        if not shutil.which("scancel"): return {"backend": "slurm", "status": "unavailable"}
        result = subprocess.run(["scancel", job_id], capture_output=True, text=True, check=False)
        return {"backend": "slurm", "status": "cancelled" if result.returncode == 0 else "failed", "output": result.stdout or result.stderr}

from .cluster import Cluster
from .models import Job, JobState, NodeState

class Scheduler:
    def __init__(self, cluster: Cluster): self.cluster = cluster; self.jobs: dict[str, Job] = {}
    def submit(self, job: Job) -> Job:
        self.jobs[job.id] = job; self.reconcile(); return job
    def reconcile(self):
        queued = sorted((j for j in self.jobs.values() if j.state == JobState.QUEUED), key=lambda j: (-j.priority, j.created_at))
        for job in queued:
            candidates = [n for n in self.cluster.nodes.values() if n.state == NodeState.READY and n.free_gpus >= job.gpu_count]
            if not candidates: continue
            node = max(candidates, key=lambda n: n.free_gpus)
            job.state = JobState.RUNNING; job.node_id = node.id
            for gpu in node.gpus[:job.gpu_count]: gpu.memory_used_mb = min(gpu.memory_total_mb, 2048)
    def cancel(self, job_id: str):
        job = self.jobs[job_id]; job.state = JobState.CANCELLED
        if job.node_id and job.node_id in self.cluster.nodes:
            for gpu in self.cluster.nodes[job.node_id].gpus: gpu.memory_used_mb = 0
        return job

    def release(self, job_id: str):
        job = self.jobs[job_id]
        if job.node_id and job.node_id in self.cluster.nodes:
            for gpu in self.cluster.nodes[job.node_id].gpus: gpu.memory_used_mb = 0
        self.reconcile()

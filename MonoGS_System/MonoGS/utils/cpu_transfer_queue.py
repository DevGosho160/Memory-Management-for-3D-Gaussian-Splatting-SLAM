"""Opt-in WSL compatibility transport for MonoGS frontend/backend messages.

torch.save copies CUDA storages through CPU bytes while retaining each tensor's
original device tag. torch.load restores the tags in the receiving process.
This avoids CUDA IPC handles; it changes transfer time, CPU memory, FPS, and
cross-process GPU allocation behavior. Do not use those as final benchmarks.
"""

import io

import torch


class CPUTransferQueue:
    def __init__(self, queue):
        self.queue = queue

    def put(self, message):
        buffer = io.BytesIO()
        torch.save(message, buffer)
        self.queue.put(buffer.getvalue())

    def get(self):
        payload = self.queue.get()
        return torch.load(io.BytesIO(payload), weights_only=False)

    def empty(self):
        return self.queue.empty()

    def close(self):
        return self.queue.close()

    def cancel_join_thread(self):
        return self.queue.cancel_join_thread()

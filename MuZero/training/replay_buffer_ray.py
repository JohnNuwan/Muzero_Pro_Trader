
import ray
from MuZero.training.replay_buffer import ReplayBuffer

@ray.remote
class ReplayBufferRay(ReplayBuffer):
    """
    Ray Actor wrapper for ReplayBuffer.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def get_buffer_size(self):
        return len(self.buffer)

from __future__ import annotations
from rreal.algorithms.rl_agent import RLAgent
import torch as th
from adarl.utils.buffers import TransitionBatch

class RandomPolicy(RLAgent):

    def __init__(self,
                 action_size : int,
                 action_min : float | list[float] = -1.0,
                 action_max : float | list[float] = 1.0,
                 torch_device : str | th.device = "cuda"):
        super().__init__()
        self._action_size = action_size
        self._action_min = th.as_tensor(action_min, device=torch_device)
        self._action_max = th.as_tensor(action_max, device=torch_device)
        self._th_device = torch_device
        self._rng = th.Generator(device = self._th_device)

    def predict_action(self, observation_batch, deterministic = False):
        return th.rand(size=(self._action_size,), generator=self._rng, device=self._th_device)*(self._action_max-self._action_min)+self._action_min
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        pass
    
    
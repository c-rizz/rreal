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
    


class FixedPolicy(RLAgent):
    def __init__(self, cmd : th.Tensor):
        self._cmd = cmd.detach().clone()

    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict = None):
        return self._cmd.clone()
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        pass

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass
    
    def load_(self, path: str):
        pass
    
    def input_device(self):
        return self._a_offset.device
    


class SinPolicy(RLAgent):
    def __init__(self,  act_scale : th.Tensor,
                        act_offset : th.Tensor,
                        act_speed : th.Tensor,
                        action_size : int,
                        dt : float,
                        verbose : bool = False):
        self._t0 = 0.0
        self._t = 0.0
        self._dt = dt
        self._a_offset = act_offset
        self._a_speed = act_speed
        # self._t_off = th.asin(self._a_offset/act_range)
        self._a_scale = act_scale.expand((action_size,))
        self._verbose = verbose

    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict = None):
        theta = (self._t0-self._t)*self._a_speed
        a = th.sin(theta)*self._a_scale+self._a_offset
        if self._verbose:
            print(f" theta = {theta} \n"
                #   f" _t_off = {self._t_off} \n"
                f" _t = {self._t} \n"
                f" _a_scale = {self._a_scale} \n"
                f" _a_offset = {self._a_offset} \n"
                f" a = {a}")
        self._t = self._t+self._dt
        return a
    
    def get_hidden_state(self):
        return self._t
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        self._t = self._t0

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass
    
    def load_(self, path: str):
        pass
    
    def input_device(self):
        return self._a_offset.device
    
    
class RandPolicy(RLAgent):
    def __init__(self,  act_scale : th.Tensor,
                        action_size : int):
        self._a_scale = act_scale.expand((action_size,))

    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict = None):
        a = (th.rand_like(self._a_scale)*2-1)*self._a_scale
        return a
    
    def get_hidden_state(self):
        return None
    
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

    def reset_hidden_state(self):
        pass

    def train_model(self, global_step, iterations, buffer: BaseBuffer) -> tuple[float, float, float]:
        raise NotImplementedError()
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass
    
    def load_(self, path: str):
        pass
    
    def input_device(self):
        return self._a_scale.device
    


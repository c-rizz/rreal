import torch.nn as nn
from adarl.utils.buffers import TransitionBatch, BaseBuffer
from abc import abstractmethod, ABC

class RLAgent(nn.Module, ABC):
    @abstractmethod
    def predict_action(self, observation_batch, deterministic = False):
        raise NotImplementedError()
    
    @abstractmethod
    def get_hidden_state(self):
        raise NotImplementedError()

    def predict(self, observation_batch, deterministic = False):
        # Mostly for stable-baselines3 compatibility
        hidden_state = self.get_hidden_state()
        return self.predict_action(observation_batch=observation_batch, deterministic=deterministic), hidden_state
    
    @abstractmethod
    def train_model(self, global_step, iterations, buffer : BaseBuffer) -> tuple[float,float,float]:
        raise NotImplementedError()

    @abstractmethod
    def reset_hidden_state(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path : str):
        raise NotImplementedError()

    @abstractmethod
    def load_(self, path : str):
        raise NotImplementedError()
    
    @abstractmethod
    def load(cls, path : str):
        raise NotImplementedError()
    
    @abstractmethod
    def input_device(self):
        raise NotImplementedError()
import torch.nn as nn
from adarl.utils.buffers import TransitionBatch
from abc import abstractmethod

class RLAgent(nn.Module):
    @abstractmethod
    def predict_action(self, observation_batch, deterministic = False):
        raise NotImplementedError()
    
    @abstractmethod
    def get_hidden_state(self):
        raise NotImplementedError

    def predict(self, observation_batch, deterministic = False):
        # Mostly for stable-baselines3 compatibility
        hidden_state = self.get_hidden_state()
        return self.predict_action(observation_batch=observation_batch, deterministic=deterministic), hidden_state
    
    @abstractmethod
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()

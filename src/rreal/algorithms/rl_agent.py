from __future__ import annotations
import zipfile
import torch.nn as nn
import yaml
from adarl.utils.buffers import TransitionBatch, BaseBuffer
from abc import abstractmethod, ABC

class RLAgent(nn.Module, ABC):
    @abstractmethod
    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict | None = None):
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
    

agents_registry = {}

def register_agent_class(agent_class : type[RLAgent], name : str | None = None):
    agents_registry[name or agent_class.__name__] = agent_class

def load_agent(path : str) -> RLAgent:
    with zipfile.ZipFile(path) as archive:
        with archive.open("extra.yaml", "r") as init_args_yamlfile:
            extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
    if not "class_name" in extra:
        raise RuntimeError(f"File does not contain class information")
    class_name = extra["class_name"]
    if not class_name in agents_registry:
        raise RuntimeError(f"Not agent registered with name '{class_name}'")
    cls = agents_registry[class_name]
    return cls.load(path)
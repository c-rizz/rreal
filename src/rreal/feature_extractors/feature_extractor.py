from __future__ import annotations
from abc import abstractmethod, ABC
import torch as th
import zipfile
import typing

class FeatureExtractor(ABC, th.nn.Module):
    @abstractmethod
    def extract_features(self, observation_batch):
        raise NotImplementedError

    @abstractmethod
    def encoding_size(self) -> int:
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def load(cls, archive : zipfile.ZipFile, name : str = "feature_extractor"):
        raise NotImplementedError()
    
    @abstractmethod
    def save_to_archive(self, archive : zipfile.ZipFile, name : str = "feature_extractor"):
        raise NotImplementedError()
    
    @abstractmethod
    def train_extractor(self, global_step, grad_steps, buffer):
        raise NotImplementedError()
    
    def get_init_args(self) -> dict[str,typing.Any]:
        return self._init_args
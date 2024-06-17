import torch.nn as nn
from adarl.utils.buffers import TransitionBatch
from abc import abstractmethod


class FeatureExtractor():
    @abstractmethod
    def extract_features(self, observation_batch):
        raise NotImplementedError

    @abstractmethod
    def encoding_size(self) -> int:
        raise NotImplementedError()
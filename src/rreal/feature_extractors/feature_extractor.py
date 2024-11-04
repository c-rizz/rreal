from abc import abstractmethod, ABC
import torch as th
import zipfile

class FeatureExtractor(ABC, th.nn.Module):
    @abstractmethod
    def extract_features(self, observation_batch):
        raise NotImplementedError

    @abstractmethod
    def encoding_size(self) -> int:
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def load(cls, archive : zipfile.ZipFile):
        raise NotImplementedError()
    
    @abstractmethod
    def save_to_archive(self, archive : zipfile.ZipFile):
        raise NotImplementedError()
    
    @abstractmethod
    def train_extractor(self, global_step, grad_steps, buffer):
        raise NotImplementedError()
    
    def get_init_args(self):
        return self._init_args
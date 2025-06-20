
from __future__ import annotations

from adarl.utils.buffers import ThDReplayBuffer, TransitionBatch, BaseBuffer, BaseValidatingBuffer
from adarl.utils.callbacks import TrainingCallback, CallbackList
from adarl.utils.tensor_trees import sizetree_from_space, map2_tensor_tree, flatten_tensor_tree, map_tensor_tree
from adarl.utils.wandb_wrapper import wandb_log
from adarl.utils.dbg.dbg_checks import dbg_check_finite
from dataclasses import dataclass, asdict
from rreal.algorithms.collectors import ExperienceCollector
from rreal.algorithms.rl_agent import RLAgent
from rreal.feature_extractors import get_feature_extractor
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor
from rreal.utils import build_mlp_net, scale_layer_weights
from typing import List, Union, Literal
import adarl.utils.callbacks
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.session
import adarl.utils.sigint_handler
import gymnasium as gym
import inspect
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import zipfile
from difflib import ndiff
import copy
from typing_extensions import override
from adarl.utils.async_cuda2cpu_queue import log_async
import pprint
import adarl.utils.spaces as spaces

class QNetwork(nn.Module):
    def __init__(self,
                 action_size : int,
                 q_network_arch : List[int],
                 observation_size : int,
                 torch_device : Union[str,th.device] = "cuda",
                 nets_num : int = 1,
                 initial_scale = 0.003):
        super().__init__()
        self._nets_num = nets_num
        self._obs_size = observation_size
        self._q_nets = build_mlp_net(arch=q_network_arch,
                                     input_size=action_size + observation_size,
                                     output_size=1,
                                     ensemble_size=self._nets_num,
                                     return_ensemble_mean=False,
                                     use_weightnorm=True,
                                     use_torchscript=True,
                                     last_layer_init_func= lambda m: scale_layer_weights(m,initial_scale)).to(device=torch_device)
    
    def get_min_qval(self, observations, actions):
        qvals = self(observations, actions)
        # ggLog.info(f"qvals.size() = {qvals.size()}")
        # min_q = qvals[:,0]
        min_q = th.min(qvals,dim=1).values # this causes a cuda sync on backward
        # min_q = min_q.squeeze(1)
        # ggLog.info(f"min_q.size() = {min_q.size()}")
        return min_q
    
    def forward(self, observations, actions):
        qvals = self._q_nets(th.cat([observations, actions], 1))
        return qvals



class Actor(nn.Module):
    def __init__(self,  action_size,
                        observation_size : int,
                        policy_arch = [256,256],
                        action_max : Union[float, th.Tensor] = 1,
                        action_min : Union[float, th.Tensor] = -1,
                        log_std_max = 2,
                        log_std_min = -5,
                        log_std_init = -3.0,
                        init_noise = 0.001,
                        torch_device : Union[str,th.device] = "cuda",
                        action_mean_init = 0.0):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self.device = torch_device
        self._obs_size = observation_size
        # save action scaling factors as non-trained parameters
        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        self.action_bias : th.Tensor
        self.action_scale : th.Tensor
        self.register_buffer("action_scale", th.as_tensor((action_max - action_min) / 2.0, dtype=th.float32, device=torch_device))
        self.register_buffer("action_bias",  th.as_tensor((action_max + action_min) / 2.0, dtype=th.float32, device=torch_device))
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.act_fc = build_mlp_net(arch=policy_arch[:-1],input_size=observation_size, output_size=policy_arch[-1],
                                    last_activation_class=th.nn.LeakyReLU).to(device=torch_device)
        mean_init = th.atanh((action_mean_init-self.action_bias)/self.action_scale).to("cpu")
        self.act_fc_mean = build_mlp_net(arch=[],
                                         input_size=policy_arch[-1],
                                         output_size=action_size,
                                         last_layer_init_func=lambda m: scale_layer_weights(m, init_noise, bias_offset=mean_init)).to(device=torch_device)
        self.act_fc_logstd = build_mlp_net(arch=[],
                                         input_size=policy_arch[-1],
                                         output_size=action_size,
                                         last_layer_init_func=lambda m: scale_layer_weights(m, init_noise, bias_offset=log_std_init)).to(device=torch_device)        

    def forward(self, observation_batch):
        hidden_batch = self.act_fc(observation_batch)
        # dbg_check_finite(hidden_batch)
        mean = self.act_fc_mean(hidden_batch)
        log_std = self.act_fc_logstd(hidden_batch)
        log_std = (th.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min # clamp the log_std network output
        return mean, log_std

    def sample_action(self, observation_batch):
        mean, log_std = self(observation_batch)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t) # squash the action in [-1,1]
        log_prob = normal.log_prob(x_t) # get the probability of the actions that we sampled

        # scale mean and action to the proper bounds
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = log_prob - th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # correct the probability for the squashing
        log_prob = log_prob.sum(1, keepdim=True) # get probability per each multidimensional action, not for each action component

        return action, log_prob, mean


class SAC(RLAgent):
    @dataclass
    class Hyperparams():
        q_lr : float
        policy_lr : float
        gamma : float
        auto_entropy_temperature : bool
        constant_entropy_temperature : float | None
        action_init : float | th.Tensor
        action_size : int
        action_min : th.Tensor
        action_max : th.Tensor
        target_tau : float
        policy_update_freq : int
        targets_update_freq : int
        q_network_arch : List[int]
        policy_arch : List[int]
        torch_device : th.device
        target_entropy : float | None
        observation_space : gym.spaces.Space
        actor_observation_space : gym.spaces.Space
        critic_observation_space : gym.spaces.Space
        feature_extractor_lr : float
        batch_size : int
        max_grad_norm : float
        log_std_init : float
        actor_observation_filter : list[str] | None
        critic_observation_filter : list[str] | None

    def __init__(self,
                 observation_space : gym.spaces.Space,
                 action_size : int,
                 q_network_arch : List[int] = [256,256],
                 q_lr : float = 0.005,
                 policy_lr : float = 0.005,
                 policy_arch : List[int] = [256,256],
                 action_min : Union[float, List[float]] = -1.0,
                 action_max : Union[float, List[float]] = 1.0,
                 torch_device : Union[str,th.device] = "cuda",
                 auto_entropy_temperature : bool = True,
                 constant_entropy_temperature : float | None = None,
                 target_entropy_factor : float | None = None,
                 gamma : float = 0.99,
                 target_tau = 0.005,
                 policy_update_freq = 2,
                 target_update_freq = 1,
                 critic_feature_extractor : FeatureExtractor | None = None,
                 actor_feature_extractor : FeatureExtractor | None = None,
                 feature_extractor_lr = 0.0,
                 batch_size = 512,
                 reference_init_args : dict = {},
                 max_grad_norm : float = 0.5,
                 actor_log_std_init = -3.0,
                 actor_observation_filter : list[str] | None = None,
                 critic_observation_filter : list[str] | None = None,
                 action_init : th.Tensor | float = 0.0):
        super().__init__()
        _, _, _, values = inspect.getargvalues(inspect.currentframe()) #type: ignore
        self._init_args = values
        self._init_args.pop("self")
        self._init_args.pop("__class__")
        self._init_args.pop("critic_feature_extractor") # Will be saved separately
        self._init_args.pop("actor_feature_extractor") # Will be saved separately
        self._init_args = copy.deepcopy(self._init_args)
        if target_entropy_factor is None:
            target_entropy_factor = -1.0
        if actor_observation_filter != None:
            if isinstance(observation_space, spaces.gym_spaces.Dict):
                actor_observation_space = spaces.gym_spaces.Dict({k:v for k,v in observation_space.spaces.items() if k in actor_observation_filter})
            else:
                raise RuntimeError(f"observation space must be a Dict to use actor_observation_filter, but it's a {type(observation_space)}")
        else:
            actor_observation_space = observation_space
        if critic_observation_filter != None:
            if isinstance(observation_space, spaces.gym_spaces.Dict):
                critic_observation_space = spaces.gym_spaces.Dict({k:v for k,v in observation_space.spaces.items() if k in critic_observation_filter})
            else:
                raise RuntimeError(f"observation space must be a Dict to use actor_observation_filter, but it's a {type(observation_space)}")
        else:
            critic_observation_space = observation_space
        self._hp = SAC.Hyperparams(q_lr=q_lr,
                                   policy_lr = policy_lr,
                                   gamma=gamma,
                                   auto_entropy_temperature=auto_entropy_temperature,
                                   constant_entropy_temperature=constant_entropy_temperature,
                                   action_init=action_init,
                                   action_size=action_size,
                                   action_min = th.as_tensor(action_min),
                                   action_max = th.as_tensor(action_max),
                                   target_tau = target_tau,
                                   policy_update_freq=policy_update_freq,
                                   targets_update_freq=target_update_freq,
                                   q_network_arch = q_network_arch,
                                   policy_arch = policy_arch,
                                   torch_device = th.device(torch_device),
                                   target_entropy = target_entropy_factor*action_size,
                                   observation_space = observation_space,
                                   feature_extractor_lr = feature_extractor_lr,
                                   batch_size = batch_size,
                                   max_grad_norm=max_grad_norm,
                                   log_std_init = actor_log_std_init,
                                   actor_observation_space = actor_observation_space,
                                   critic_observation_space = critic_observation_space,
                                   actor_observation_filter = actor_observation_filter,
                                   critic_observation_filter = critic_observation_filter)
        self._obs_space_sizes = sizetree_from_space(observation_space)
        self.device = self._hp.torch_device
        self._critic_updates = 0
        self._alpha_updates = 0
        self._policy_updates = 0
        self._share_actor_critic_feature_extractor = (actor_feature_extractor==critic_feature_extractor and
                                                      actor_observation_filter==critic_observation_filter)
        if self._share_actor_critic_feature_extractor:
            if critic_feature_extractor is None or actor_feature_extractor is None: # second considition is just for typing
                self._critic_feature_extractor = StackVectorsFeatureExtractor(observation_space=critic_observation_space,
                                                                   device=self._hp.torch_device)
                self._actor_feature_extractor = self._critic_feature_extractor
            else:
                self._critic_feature_extractor = critic_feature_extractor
                self._actor_feature_extractor = actor_feature_extractor
        else:
            if critic_feature_extractor is None:
                self._critic_feature_extractor = StackVectorsFeatureExtractor(observation_space=critic_observation_space,
                                                                    device=self._hp.torch_device)
            else:
                self._critic_feature_extractor = critic_feature_extractor
            if actor_feature_extractor is None:
                self._actor_feature_extractor = StackVectorsFeatureExtractor(observation_space=actor_observation_space,
                                                                    device=self._hp.torch_device)
            else:
                self._actor_feature_extractor = actor_feature_extractor
        self._q_net = QNetwork( observation_size=self._critic_feature_extractor.encoding_size(),
                                action_size=self._hp.action_size,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device,
                                nets_num=2)
        self._q_net_target = QNetwork(  observation_size=self._critic_feature_extractor.encoding_size(),
                                        action_size=self._hp.action_size,
                                        q_network_arch=q_network_arch,
                                        torch_device=self._hp.torch_device,
                                        nets_num=2)
        self._q_net_target.load_state_dict(self._q_net.state_dict())
        self._q_optimizer = optim.Adam(self._q_net.parameters(), lr=self._hp.q_lr)
        self._actor = Actor(policy_arch=policy_arch,
                            observation_size=self._actor_feature_extractor.encoding_size(),
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.torch_device,
                            log_std_init=self._hp.log_std_init,
                            action_mean_init=self._hp.action_init)
        self._actor_optimizer = optim.Adam(list(self._actor.parameters()), lr=self._hp.policy_lr)
        if self._hp.auto_entropy_temperature:
            self._target_entropy = self._hp.target_entropy
            self._log_alpha = th.zeros(1, requires_grad=True, device=torch_device)
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = th.as_tensor(constant_entropy_temperature).to(device=self._hp.torch_device, non_blocking=self._hp.torch_device.type=="cuda")

        if self._hp.feature_extractor_lr > 0:
            self._critic_feature_extractor_optimizer = optim.Adam(list(self._critic_feature_extractor.parameters()), lr=self._hp.feature_extractor_lr)
            if self._share_actor_critic_feature_extractor:
                self._actor_feature_extractor_optimizer = self._critic_feature_extractor_optimizer
            else:
                self._actor_feature_extractor_optimizer = optim.Adam(list(self._actor_feature_extractor.parameters()), lr=self._hp.feature_extractor_lr)
        else:
            self._critic_feature_extractor_optimizer = None
            self._actor_feature_extractor_optimizer = None

        self._last_q_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_actor_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_alpha_loss = th.as_tensor(float("nan"), device=self.device)
        self._tot_grad_steps_count = 0
        self._stats = { "tot_grad_steps_count":0,
                        "q_loss":0.0,
                        "actor_loss":0.0,
                        "alpha_loss":0.0,
                        "val_q_loss":0.0,
                        "val_actor_loss":0.0,
                        "val_alpha_loss":0.0,
                        "alpha":0.0}

    def get_actor_subobservation(self, observation : dict | th.Tensor):
        if self._hp.actor_observation_filter is None:
            return observation
        else:
            r = {k:observation.get(k,None) for k in self._hp.actor_observation_filter}
            return {k:v for k,v in r.items() if v is not None}

    def get_critic_subobservation(self, observation : dict | th.Tensor):
        if self._hp.critic_observation_filter is None:
            return observation
        else:
            r = {k:observation.get(k,None) for k in self._hp.critic_observation_filter}
            return {k:v for k,v in r.items() if v is not None}

    def get_feature_extractors(self):
        return self._critic_feature_extractor, self._actor_feature_extractor

    def save(self, path : str):
        with zipfile.ZipFile(path, mode="w") as archive:
            with archive.open("sac.pth", "w") as sac_file:
                th.save(self.state_dict(), sac_file)
            with archive.open("extra.yaml", "w") as extra_file:
                extra = {}
                extra["init_args"] = self._init_args
                extra["hyperparams"] = asdict(self._hp)
                extra["class_name"] = self.__class__.__name__
                extra["critic_feature_extractor_class_name"] = self._critic_feature_extractor.__class__.__name__
                extra["critic_feature_extractor_init_args"] = self._critic_feature_extractor.get_init_args()
                extra["actor_feature_extractor_class_name"] = self._actor_feature_extractor.__class__.__name__
                extra["actor_feature_extractor_init_args"] = self._actor_feature_extractor.get_init_args()
                extra["share_feature_extractor"] = self._share_actor_critic_feature_extractor
                # print(extra)
                # for k in extra["init_args"]:
                #     print("k=",k)
                #     yaml.dump(extra["init_args"][k],default_flow_style=None)
                extra_file.write(yaml.dump(extra,default_flow_style=None).encode("utf-8"))
            self._critic_feature_extractor.save_to_archive(archive, name="critic_feature_extractor")
            self._actor_feature_extractor.save_to_archive(archive, name="actor_feature_extractor")
            # th.save( self._feature_extractor.state_dict(), path+".fe_state.pth")
        
    def _check_feature_extractor(self, current_featur_extractor, loaded_fe_name, loaded_fe_args):
        if current_featur_extractor.__class__.__name__ != loaded_fe_name:
            ggLog.warn(f"feature_extractor_class_name of loaded model differs from that of self.\n"
                       f"loaded = {loaded_fe_name}, self's = {self._critic_feature_extractor.__class__.__name__}")
            raise RuntimeError("Unmatched init_args")
        if current_featur_extractor.get_init_args() != loaded_fe_args:
            import difflib
            self_init_args_yaml = yaml.dump(current_featur_extractor.get_init_args())
            load_init_args_yaml = yaml.dump(loaded_fe_args)
            diff = "".join(difflib.unified_diff(self_init_args_yaml.splitlines(keepends=True),
                                        load_init_args_yaml.splitlines(keepends=True),
                                        fromfile="self",
                                        tofile="loaded",
                                        lineterm=""))
            ggLog.warn(f"init args of loaded model differ from those of self.\n"
                       f"self init_args = \n{self_init_args_yaml}\n"
                       f"load init_args = \n{load_init_args_yaml}\n"
                       f"diff init_args = \n{diff}")
            raise RuntimeError("Unmatched init_args")

    def load_(self, path : str):
        # Before loading the state dict we try to check that the models are compatible
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != self.__class__.__name__:
            raise RuntimeError(f"File was not saved by this class")
        if self._init_args != extra["init_args"]:
            ggLog.warn("init args of loaded model differ from those of self.")
            load_yaml_args = yaml.dump(extra['init_args'])
            original_yaml_args = yaml.dump(self._init_args)
            ggLog.warn(f"self._init_args = \n{original_yaml_args}")
            ggLog.warn(f"load init_args  = \n{load_yaml_args}")
            diffs = ndiff(   original_yaml_args.splitlines(keepends=True),
                            load_yaml_args.splitlines(keepends=True))
            diffs = [l for l in diffs if len(l)>0 and l[0] != ' ']
            ggLog.warn(f"Args comparison with loaded model:\n{''.join(diffs)}")
            # raise RuntimeError("Unmatched init_args")
        self._check_feature_extractor(self._critic_feature_extractor,
                                      extra["critic_feature_extractor_class_name"],
                                      extra["critic_feature_extractor_init_args"])
        self._check_feature_extractor(self._actor_feature_extractor,
                                      extra["actor_feature_extractor_class_name"],
                                      extra["actor_feature_extractor_init_args"])
        with zipfile.ZipFile(path) as archive:
            with archive.open("sac.pth", "r") as sac_file:
                self.load_state_dict(th.load(sac_file))

    @classmethod
    def load(cls, path : str):
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class")
        sac_init_args = extra["init_args"]
        with zipfile.ZipFile(path) as archive:
            critic_feature_extractor_class = get_feature_extractor(extra["critic_feature_extractor_class_name"])
            sac_init_args["critic_feature_extractor"] = critic_feature_extractor_class.load(archive, name="critic_feature_extractor")
            if extra["share_feature_extractor"]:
                sac_init_args["actor_feature_extractor"] = sac_init_args["critic_feature_extractor"]
            else:
                actor_feature_extractor_class = get_feature_extractor(extra["actor_feature_extractor_class_name"])
                sac_init_args["actor_feature_extractor"] = actor_feature_extractor_class.load(archive, name="actor_feature_extractor")
        ggLog.info(f"load(): building model with args: \n"+pprint.pformat(sac_init_args))
        model = SAC(**sac_init_args)
        # At this point we should have a model that is initialized exactly like the one that was saved
        # So we can load into it the state from the checkpoint
        model.load_(path)
        return model

    @override
    def predict_action(self, observation_batch, deterministic = False):
        # s = {k:v.size() for k,v in observation.items()}
        # ggLog.info(f"predict: observation = {s}")


        # check if it is not a batch, if so, unsqueeze
        has_right_dims = map2_tensor_tree(observation_batch, self._obs_space_sizes,
                                            lambda obs,obs_space_size: obs.dim() == len(obs_space_size))
        is_batched = not all(flatten_tensor_tree(has_right_dims).values())
        if not is_batched:
            observation_batch = map_tensor_tree(observation_batch, lambda t: t.unsqueeze(0))

        observation_batch = self.get_actor_subobservation(observation_batch)
        observation_batch = map_tensor_tree(observation_batch, lambda t: t.to(device = self.device, dtype = th.float32))
        observation_batch = self._actor_feature_extractor.extract_features(observation_batch)
        action, log_prob, mean = self._actor.sample_action(observation_batch)
        if not is_batched:
            action = action.squeeze()
            mean = mean.squeeze()
            log_prob = log_prob.squeeze()
        if deterministic:
            return mean
        else:
            return action

    @override
    def get_hidden_state(self):
        return None

    @override
    def reset_hidden_state(self):
        return

    def _compute_critic_loss(self, transitions : TransitionBatch):
        critic_obss = self.get_critic_subobservation(transitions.observations)
        critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss)
        with th.no_grad():
            critic_next_obss = self.get_critic_subobservation(transitions.next_observations)
            crit_next_enc_obss = self._critic_feature_extractor.extract_features(critic_next_obss)   
            if self._share_actor_critic_feature_extractor:     
                act_next_enc_obss = crit_next_enc_obss
            else:
                actor_next_obss = self.get_actor_subobservation(transitions.next_observations)
                act_next_enc_obss = self._actor_feature_extractor.extract_features(actor_next_obss)
            # Compute next-values for TD
            next_state_actions, next_state_log_pi, _ = self._actor.sample_action(act_next_enc_obss)
            q_next = self._q_net_target.get_min_qval(crit_next_enc_obss, next_state_actions)
            soft_q_next = q_next - self._alpha * next_state_log_pi
            td_q_values = transitions.rewards.flatten() + (1 - transitions.terminated.flatten()) * self._hp.gamma * (soft_q_next).view(-1)

        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        q_values = self._q_net(critic_enc_obss, transitions.actions)
        # ggLog.info(f"q_values.size() = {q_values.size()}")
        td_q_values = td_q_values.unsqueeze(1).unsqueeze(2)
        td_q_values = td_q_values.expand(-1,2,1)
        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        return F.mse_loss(q_values, td_q_values)

    def _update_critic(self, transitions : TransitionBatch):
        q_loss = self._compute_critic_loss(transitions)
        self._q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self._q_net.parameters(), self._hp.max_grad_norm)
        self._q_optimizer.step()
        self._critic_updates += 1
        self._last_q_loss = q_loss.detach()


    def _compute_actor_loss(self, transitions : TransitionBatch):
        actor_obss = self.get_actor_subobservation(transitions.observations)
        actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss)
        act, act_log_prob, _ = self._actor.sample_action(actor_enc_obss)
        # with th.no_grad():
        if self._share_actor_critic_feature_extractor:
            critic_enc_obss = actor_enc_obss
        else:
            critic_obss = self.get_critic_subobservation(transitions.observations)
            critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss)
        min_q_pi = self._q_net.get_min_qval(critic_enc_obss, act) # cannot reuse those from _update_value_func, the value function has changed
        # ggLog.info(f"min_q_pi.size() = {min_q_pi.size()}")
        # ggLog.info(f"act_log_prob.size() = {act_log_prob.size()}")
        return ((self._alpha * act_log_prob) - min_q_pi).mean()

    def _update_actor(self, transitions : TransitionBatch):
        actor_loss = self._compute_actor_loss(transitions)
        self._actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self._hp.max_grad_norm)
        self._actor_optimizer.step()
        self._last_actor_loss = actor_loss.detach()
        self._policy_updates += 1

    def _compute_alpha_loss(self, transitions : TransitionBatch):
        with th.no_grad():
            actor_obss = self.get_actor_subobservation(transitions.observations)
            actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss)
            _, act_log_prob, _ = self._actor.sample_action(actor_enc_obss)
            self._stats["avg_log_prob"] = act_log_prob.mean()
        return (-self._log_alpha.exp() * (act_log_prob + self._target_entropy)).mean()
    
    def _update_alpha(self, transitions : TransitionBatch):
        if self._hp.auto_entropy_temperature:
            alpha_loss = self._compute_alpha_loss(transitions)
            self._alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self._log_alpha, self._hp.max_grad_norm)
            self._alpha_optimizer.step()
            self._alpha = self._log_alpha.exp().detach()
        else:
            alpha_loss = th.tensor(0.0, device=self.device)
        self._last_alpha_loss = alpha_loss.detach()
        self._alpha_updates += 1

    @staticmethod
    def _target_update(param, target_param, tau):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
    def _update_target_nets(self):
        for param, target_param in zip(self._q_net.parameters(), self._q_net_target.parameters()):
            self._target_update(param, target_param, self._hp.target_tau)

    def _update_feature_extractor(self):
        if self._share_actor_critic_feature_extractor:
            if self._critic_feature_extractor_optimizer is not None:
                self._critic_feature_extractor_optimizer.step()
        else:
            if self._critic_feature_extractor_optimizer is not None:
                self._critic_feature_extractor_optimizer.step()
            if self._actor_feature_extractor_optimizer is not None:
                self._actor_feature_extractor_optimizer.step()

    def _update(self, transitions : TransitionBatch):
        # sync_dbg_mode = th.cuda.get_sync_debug_mode()
        # th.cuda.set_sync_debug_mode("error")
        if self._critic_feature_extractor_optimizer is not None:
            self._critic_feature_extractor_optimizer.zero_grad(set_to_none=True)
        if self._actor_feature_extractor_optimizer is not None:
            self._actor_feature_extractor_optimizer.zero_grad(set_to_none=True)
        self._update_critic(transitions = transitions)
        did_train_something = False
        if self._critic_updates % self._hp.policy_update_freq == 0:
            for _ in range(self._hp.policy_update_freq):
                self._update_actor(transitions=transitions)
                self._update_alpha(transitions=transitions)
                did_train_something = True
        if self._critic_updates % self._hp.targets_update_freq == 0:
            self._update_target_nets()
            did_train_something = True
        if did_train_something:
            self._update_feature_extractor()
        # th.cuda.set_sync_debug_mode(sync_dbg_mode)
        return self._last_q_loss, self._last_actor_loss, self._last_alpha_loss
    
    def validate(self, buffer : BaseValidatingBuffer, batch_size : int):
        with th.no_grad():
            transitions = buffer.sample_validation(batch_size=batch_size)
            critic_loss = self._compute_critic_loss(transitions)
            actor_loss = self._compute_actor_loss(transitions)
            alpha_loss = self._compute_alpha_loss(transitions)
        self._stats.update({"val_q_loss":critic_loss,
                            "val_actor_loss":actor_loss,
                            "val_alpha_loss":alpha_loss})
        return critic_loss, actor_loss, alpha_loss

    @override
    def train_model(self, global_step, iterations, buffer : BaseBuffer) -> tuple[th.Tensor,th.Tensor,th.Tensor]:
        q_act_alpha_losses = [None]*iterations
        for i in range(iterations):
            transitions = buffer.sample(self._hp.batch_size)
            transitions = map_tensor_tree(transitions, lambda t : t.to(device=self.device, non_blocking=True))
            # th.cuda.synchronize(self.device)
            q_act_alpha_losses[i] = self._update(transitions = transitions)
            self._tot_grad_steps_count += 1
        # q_loss, actor_loss, alpha_loss = th.as_tensor(q_act_alpha_losses).mean(dim = 0).cpu().numpy()
        q_loss, actor_loss, alpha_loss = q_act_alpha_losses[-1]
        adarl.utils.session.default_session.run_info["train_iterations"].value = self._tot_grad_steps_count
        self._stats.update({"tot_grad_steps_count":self._tot_grad_steps_count,
                            "q_loss":q_loss,
                            "actor_loss":actor_loss,
                            "alpha_loss":alpha_loss,
                            "alpha":self._alpha})
        return q_loss, actor_loss, alpha_loss

    @override
    def input_device(self):
        return self._hp.torch_device
    
    def get_stats(self):
        return self._stats

def train_off_policy(collector : ExperienceCollector,
                    model : SAC,
                    buffer : BaseBuffer,
                    total_timesteps : int,
                    train_freq : int,
                    learning_start_step : int,
                    grad_steps : int | Literal["auto"],
                    log_freq_vstep : int = -1,
                    callbacks : Union[TrainingCallback, List[TrainingCallback]] | None = None,
                    validation_freq : int = 1,
                    validation_batch_size : int = 256):
    if validation_freq>0 and not isinstance(buffer, BaseValidatingBuffer):
        raise RuntimeError(f"validation_freq>0 but buffer is not a BaseValidatingBuffer")
    if log_freq_vstep == -1: log_freq_vstep = train_freq
    num_envs = collector.num_envs()

    collector.reset()
    global_step = 0
    t_coll_sl = 0
    t_train_sl = 0
    t_val_sl = 0
    t_buff_sl = 0
    t_tot_sl = 0
    t_add_sl = 0
    
    if callbacks is None:
        callbacks = []
    if not isinstance(callbacks, CallbackList):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]    
        callbacks = CallbackList(callbacks=callbacks)

    callbacks.on_training_start()
    ep_counter = 0
    step_counter = 0
    steps_sl = 0
    train_count = 0
    q_loss, actor_loss, alpha_loss = th.as_tensor(float("nan")),th.as_tensor(float("nan")),th.as_tensor(float("nan"))
    start_time = time.monotonic()
    last_log_steps = float("-inf")


    # th.cuda.memory._record_memory_history(max_entries=100_000)
    while global_step < total_timesteps and not adarl.utils.session.default_session.is_shutting_down():
        s0b = buffer.collected_frames()
        t0 = time.monotonic()

        # ------------------  Start experience collection  ------------------
        steps_to_collect = train_freq*num_envs
        vsteps_to_collect = train_freq
        callbacks.on_collection_start()
        collector.start_collection(model_state_dict=model.state_dict(),
                                            vsteps_to_collect=vsteps_to_collect,
                                            global_vstep_count=global_step//num_envs,
                                            random_vsteps=learning_start_step//num_envs)

        # ------------------             Train             ------------------
        t_before_train = time.monotonic()
        trained = False
        if global_step > learning_start_step:
            while (grad_steps != "auto" and not trained) or (grad_steps == "auto" and collector.is_collecting()):
                trained = True
                q_loss, actor_loss, alpha_loss = model.train_model(global_step, grad_steps if grad_steps!="auto" else 10, buffer)
            train_count += 1
        t_after_train = time.monotonic()
        if trained and validation_freq>0 and train_count%validation_freq==0:
            model.validate(buffer, batch_size=validation_batch_size)
        t_after_val = time.monotonic()
        if trained:
            # ggLog.info(f"SAC: "+str([f"{k}={v}, " for k,v in model.get_stats().items()]))
            wlogs = {"sac/"+k:v for k,v in model.get_stats().items()}
            wlogs["sac/buffer_frames"] = buffer.stored_frames()
            wlogs["sac/val_buffer_frames"] = buffer.stored_validation_frames() if isinstance(buffer,BaseValidatingBuffer) else 0
            wandb_log(wlogs,throttle_period=2, silent_throttling=True)
        adarl.utils.session.default_session.run_info["train_iterations"].value = model._tot_grad_steps_count
        
        # ------------------   Store collected experience  ------------------
        tmp_buff = collector.wait_collection(timeout = 300.0)
        new_episodes = tmp_buff.added_completed_episodes() - ep_counter
        ep_counter = tmp_buff.added_completed_episodes()
        step_counter = tmp_buff.added_frames()
        t_coll_sl += collector.collection_duration()
        adarl.utils.session.default_session.run_info["collected_episodes"].value = ep_counter
        adarl.utils.session.default_session.run_info["collected_steps"].value = step_counter
        # callbacks._callbacks[0].set_model(model)
        callbacks.on_collection_end(collected_steps=vsteps_to_collect*num_envs,
                                   collected_episodes=new_episodes,
                                   collected_data=tmp_buff)
        t_before_buff = time.monotonic()
        t_add = 0
        for (obs, next_obs, action, reward, terminated, truncated) in tmp_buff.replay():
            tpa = time.monotonic()
            buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward,
                        truncated=truncated, terminated=terminated)
            t_add += time.monotonic() - tpa
        t_after_buff = time.monotonic()

        # ------------------      Wrap up and restart      ------------------
        if buffer.collected_frames()-s0b != steps_to_collect:
            raise RuntimeError(f"Expected to collect {steps_to_collect} but got {buffer.stored_frames()-s0b}")
        global_step += steps_to_collect
        steps_sl += steps_to_collect
        tf = time.monotonic()
        t_train_sl += t_after_train - t_before_train
        t_val_sl += t_after_val - t_after_train
        t_buff_sl += t_after_buff - t_before_buff
        t_add_sl += t_add
        t_tot_sl += tf-t0
        t = time.monotonic()
        # ggLog.info(f"global_steps = {global_step}")
        if global_step - last_log_steps > log_freq_vstep*num_envs:
            last_log_steps = global_step
            log_async(f"SAC: expsteps={global_step}"+" q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}",
                      tensors=dict(q_loss=q_loss,actor_loss=actor_loss,alpha_loss=alpha_loss))
            # ggLog.info(f"SAC: expsteps={global_step} q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}")
            ggLog.info(f"OFFTRAIN: expstps:{global_step}"
                       f" trainstps={model._tot_grad_steps_count}"
                    #    f" exp_reuse={model._tot_grad_steps_count*batch_size/global_step:.2f}"
                       f" coll={t_coll_sl:.2f}s train={t_train_sl:.2f}s val={t_val_sl:.2f}s buff={t_buff_sl:.2f}s add={t_add_sl:.2f}s tot={t_tot_sl:.2f}"
                       f" fps={steps_sl/t_tot_sl:.2f} collfps={steps_sl/t_coll_sl:.2f}"
                       f" alltime_fps={global_step/(t-start_time):.2f} alltime_ips={model._tot_grad_steps_count/(t-start_time):.2f}")
            dictlist = [f"{k}:{v:.6g}" for k,v in collector.get_stats().items()]
            ggLog.info(f"Collection: {', '.join(dictlist)}")
            t_train_sl, t_coll_sl, t_tot_sl, steps_sl, t_val_sl, t_buff_sl, t_add_sl = 0,0,0,0,0,0,0
            free, total = th.cuda.mem_get_info(th.device('cuda:0'))
            mem_used_MB = (total - free) / 1024 ** 2
            # ggLog.info(f"{t}: cuda mem usage = {mem_used_MB}")
        # jax.profiler.save_device_memory_profile(f"jax_memory_{t}.prof")
        # th.cuda.memory._dump_snapshot(f"memory_{t}_th.pickle")
        adarl.utils.sigint_handler.haltOnSigintReceived()
    callbacks.on_training_end()

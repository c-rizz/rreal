# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import math
import os
import random
import time
import dataclasses
from dataclasses import dataclass, asdict
from tracemalloc import start
import yaml
import zipfile
import pprint

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from rreal.algorithms.sac_helpers import gym_builder, build_vec_env, wrap_with_logger, env_builder2vec, build_eval_callbacks
import adarl.utils.session
import inspect
import copy
from rreal.algorithms.sac_helpers import EnvBuilderProtocol, VecEnvBuilderProtocol
from typing import Any, Final, TypeVar
from adarl.utils.callbacks import TrainingCallback, CallbackList, CheckpointCallbackRB
import adarl.utils.sigint_handler
from rreal.algorithms.rl_agent import RLAgent, register_agent_class
from rreal.algorithms.sac import compare_dicts
from rreal.feature_extractors import get_feature_extractor
from typing_extensions import override
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor, StackVectorsFeatureExtractorInitArgs
from adarl.utils.utils import numpy_to_torch_dtype_dict, get_func_input_args, override_struct
from adarl.utils.tensor_trees import map_tensor_tree
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
from adarl.utils.wandb_wrapper import wandb_log
from rreal.utils.utils import build_mlp_net
import jax.profiler
from rreal.utils.FixedAdamW import AdamW
from rreal.utils.utils import simplified_clip_grad_norm_
import wandb
from adarl.utils.tensor_trees import TensorTree
import hdf5plot.save
import adarl.utils.spaces as spaces

th._dynamo.config.compiled_autograd = True

def layer_init(layer, std=2.0, bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

def ortho_layer_init_(layer, std=2.0, bias_const : th.Tensor | float = 0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, 0.0)
    layer.bias += bias_const.to(device=layer.bias.device) if isinstance(bias_const, th.Tensor) else bias_const


class PPORolloutBuffer():
    def __init__(self, num_steps, num_envs, obs_space : spaces.gym_spaces.Dict, act_space_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self._storage_th_device = device
        self._obs = {
            key: th.zeros(  size=(num_steps+1, num_envs) + space.shape,
                            dtype=numpy_to_torch_dtype_dict[space.dtype],
                            device = self._storage_th_device)
            for key, space in obs_space.spaces.items()
        }
        self._actions = th.zeros((num_steps, num_envs) + act_space_shape, device=device)
        self._logprobs = th.zeros((num_steps+1, num_envs), device=device)
        self._rewards = th.zeros((num_steps, num_envs), device=device)
        self._terminateds = th.zeros((num_steps+1, num_envs), device=device)
        self._truncateds = th.zeros((num_steps+1, num_envs), device=device)
        self._values = th.zeros((num_steps+1, num_envs), device=device)
        self._consequent_values = th.zeros((num_steps+1, num_envs), device=device)
        self._pos = 0
        tot_bytes = self._actions.nbytes + self._logprobs.nbytes + self._rewards.nbytes + self._terminateds.nbytes + self._truncateds.nbytes + self._values.nbytes + self._consequent_values.nbytes
        ggLog.info(f"PPO Rollout buffer will occupy {tot_bytes/1024/1024}MiB")

    @th.compile(fullgraph=True, mode="max-autotune")
    def add(self, start_obss, actions, logprobs, rewards, prev_terminateds, prev_truncateds, start_obss_values, consequent_obss_values):
        for k in start_obss.keys():
            self._obs[k][self._pos] = start_obss[k]
        self._actions[self._pos] = actions
        self._rewards[self._pos] = rewards
        self._terminateds[self._pos] = prev_terminateds
        self._truncateds[self._pos] = prev_truncateds
        self._values[self._pos] = start_obss_values
        self._consequent_values[self._pos] = consequent_obss_values
        self._logprobs[self._pos] = logprobs
        self._pos += 1

    def set(self, start_obss=None, actions=None, logprobs=None, rewards=None, prev_terminateds=None, prev_truncateds=None, start_obss_values=None,
                consequent_obss_values=None):
        if start_obss is not None:
            for k in start_obss.keys():
                self._obs[k][self._pos] = start_obss[k]
        if actions is not None:
            self._actions[self._pos] = actions
        if rewards is not None:
            self._rewards[self._pos] = rewards
        if prev_terminateds is not None:
            self._terminateds[self._pos] = prev_terminateds
        if prev_truncateds is not None:
            self._truncateds[self._pos] = prev_truncateds
        if start_obss_values is not None:
            self._values[self._pos] = start_obss_values
        if consequent_obss_values is not None:
            self._consequent_values[self._pos] = consequent_obss_values
        if logprobs is not None:
            self._logprobs[self._pos] = logprobs

    def reset(self):
        self._pos = 0

    def get_rollout_data(self):
        return (self._obs, #[:self._pos+1],
                self._actions, #[:self._pos],
                self._rewards, #[:self._pos],
                self._terminateds, #[:self._pos+1],
                self._truncateds, #[:self._pos+1],
                self._logprobs, #[:self._pos+1])
                self._values,
                self._consequent_values) #[:self._pos+1],


# # From https://github.com/pytorch/pytorch/issues/79197#issuecomment-1434511798
# from dataclasses import is_dataclass
# from typing import TypeVar
# from torch.jit._dataclass_impls import synthesize__init__
# import re
# import inspect
# import tempfile
# import importlib.util
# import importlib.machinery
# import torch
# T = TypeVar("T", bound=type)
# # TODO: support __eq__ and __repr__
# def jittable(cls: T) -> T:
#     assert is_dataclass(cls)
#     src = synthesize__init__(cls).source.replace(f"{cls.__module__}.{cls.__qualname__}", cls.__name__)
#     # get `globals()` from the caller
#     globals_dict = {k: v for k, v in inspect.stack()[1][0].f_globals.items()
#                     if not re.match(r"__\w+__", k)}
#     # # This is to handle `from ... import`, where the imported names are already in `globals()`
#     # for param in inspect.signature(cls.__init__).parameters.values():
#     #     if param.annotation.__module__.split(".")[0] not in globals_dict:
#     #         src = src.replace(f"{param.annotation.__module__}.", "")
#     # write source code into a temp file to allow `inspect.getsource` to load the source code
#     with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
#         tmp.write(src)
#         tmp.flush()
#     loader = importlib.machinery.SourceFileLoader(cls.__name__, tmp.name)
#     spec = importlib.util.spec_from_loader(loader.name, loader)
#     module = importlib.util.module_from_spec(spec)
#     for k, v in globals_dict.items():
#         setattr(module, k, v)
#     setattr(module, cls.__name__, cls)
#     spec.loader.exec_module(module)
#     cls.__init__ = module.__init__
#     # This is to prevent `jit.script` from recursively scripting annotations
#     cls.__annotations__.clear()
#     if not issubclass(cls, nn.Module):
#         # Let `@jit.script` treat `cls` as a normal class
#         del cls.__dataclass_fields__
#     # Write `cls` back to caller's `globals()`, so that nested classes are visible to `@jit.script`
#     inspect.stack()[1][0].f_globals[cls.__name__] = cls
#     return cls

@dataclass(repr=False, eq=False)
class PPO_Hyperparams: # We keep it outside ppo because yaml does not handle nested classes https://github.com/yaml/pyyaml/issues/131
    minibatch_size: int | None
    minibatch_num: int | None
    th_device : th.device
    action_space : spaces.ThBox
    observation_space : spaces.gym_spaces.Space
    action_min : th.Tensor
    action_max : th.Tensor
    q_lr : float
    policy_lr : float
    num_envs: int
    num_steps: int
    update_epochs: int
    # anneal_lr: bool = True
    # """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    epsilon_value_clip_epsilon: float = 0.2
    """the clipping coefficient used in the value function GAE loss"""
    epsilon_policy_ratio_clip: float = 0.2
    """The clipping coefficient used in the policy gradient loss, expressed as the maximum allowed ratio between new and old policy"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    loss_entropy_weight: float = 0.001
    """coefficient of the entropy"""
    loss_value_weight: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    critic_network_arch : tuple[int,...] = (64,64)
    """The layer sizes of the critic MLP"""
    actor_network_arch : tuple[int,...] = (64,64)
    """The layer sizes of the actor MLP"""
    actor_observation_filter : list[str] | None = None
    """Subset of observation keys visible to the actor. Requires Dict observations."""
    critic_observation_filter : list[str] | None = None
    """Subset of observation keys visible to the critic. Requires Dict observations."""
    init_actor_logstd : float = 0.0
    """The initial value for the actor log standard deviation parameters"""
    actor_mean_bounds_ratio : float = 1.0
    """Fraction of the post-tanh action range that the actor's mean is allowed to reach.
    With 1.0 (default) the mean can reach the action bounds; reducing it (e.g. 0.9) keeps the mean
    away from ±1 so the tanh squashing does not eat the action noise asymmetrically near the
    boundaries."""
    reference_init_args : dict = dataclasses.field(default_factory=dict)
    """Additional arguments that will be saved together with the model, just for reference on how it was trained"""
    

class PPO(RLAgent):


    _hp : Final[PPO_Hyperparams]

    def _get_filtered_observation_space(self,
                                        observation_space : spaces.gym_spaces.Space,
                                        obs_filter : list[str] | None,
                                        role : str) -> spaces.gym_spaces.Space:
        if obs_filter is None:
            return observation_space
        if not isinstance(observation_space, spaces.gym_spaces.Dict):
            raise RuntimeError(f"observation space must be a Dict to use {role}_observation_filter, but it's a {type(observation_space)}")
        return spaces.ThDict({k:v for k,v in observation_space.spaces.items() if k in obs_filter})

    def __init__(self, hyperparams : PPO_Hyperparams,
                 feature_extractor : FeatureExtractor | None = None,
                 actor_feature_extractor : FeatureExtractor | None = None,
                 critic_feature_extractor : FeatureExtractor | None = None):
        super().__init__()
        self._init_args = get_func_input_args(exclude=[ "self",
                                                        "values",
                                                        "__class__",
                                                        "feature_extractor",
                                                        "critic_feature_extractor",
                                                        "actor_feature_extractor"])
        self._init_args = copy.deepcopy(self._init_args)
        self._hp = copy.deepcopy(hyperparams)
        action_mean_init=self._hp.action_space.zero_action.to(device=self._hp.th_device) if isinstance(self._hp.action_space,spaces.ThBox) else 0.0
        self._action_len = int(np.prod(self._hp.action_space.shape))
        self._actor_observation_space = self._get_filtered_observation_space(self._hp.observation_space,
                                                                             self._hp.actor_observation_filter,
                                                                             "actor")
        self._critic_observation_space = self._get_filtered_observation_space(self._hp.observation_space,
                                                                              self._hp.critic_observation_filter,
                                                                              "critic")
        ggLog.info(f"PPO Actor observation space labels: "+str({k:v.labels for k,v in self._actor_observation_space.spaces.items()}))
        ggLog.info(f"PPO Critic observation space labels: "+str({k:v.labels for k,v in self._critic_observation_space.spaces.items()}))
        time.sleep(5)
        if feature_extractor is not None and (actor_feature_extractor is not None or critic_feature_extractor is not None):
            raise RuntimeError("Provide either feature_extractor or actor/critic_feature_extractor, not both.")
        same_observation_filter = self._hp.actor_observation_filter == self._hp.critic_observation_filter
        if feature_extractor is not None:
            if not same_observation_filter:
                raise RuntimeError("Cannot share a single feature_extractor when actor and critic observation filters differ.")
            self._actor_feature_extractor = feature_extractor
            self._critic_feature_extractor = feature_extractor
        else:
            if actor_feature_extractor is None and critic_feature_extractor is None and same_observation_filter:
                shared_extractor = StackVectorsFeatureExtractor(observation_space=self._actor_observation_space,
                                                                hp=StackVectorsFeatureExtractorInitArgs(device=hyperparams.th_device))
                self._actor_feature_extractor = shared_extractor
                self._critic_feature_extractor = shared_extractor
            else:
                if critic_feature_extractor is None:
                    critic_feature_extractor = StackVectorsFeatureExtractor(observation_space=self._critic_observation_space,
                                                                            hp=StackVectorsFeatureExtractorInitArgs(device=hyperparams.th_device))
                if actor_feature_extractor is None:
                    actor_feature_extractor = StackVectorsFeatureExtractor(observation_space=self._actor_observation_space,
                                                                            hp=StackVectorsFeatureExtractorInitArgs(device=hyperparams.th_device))
                self._critic_feature_extractor = critic_feature_extractor
                self._actor_feature_extractor = actor_feature_extractor
        self._share_actor_critic_feature_extractor = self._actor_feature_extractor is self._critic_feature_extractor
        self.critic = build_mlp_net(arch=self._hp.critic_network_arch,
                                    input_size=self._critic_feature_extractor.encoding_size(),
                                    output_size=1,
                                    # use_weightnorm=True,
                                    use_torchscript=True,
                                    # hidden_activations=th.nn.Tanh,
                                    layer_init_func=lambda m: ortho_layer_init_(m,1**0.5),
                                    last_layer_init_func=lambda m: ortho_layer_init_(m,0.01)).to(device=self._hp.th_device)
        self.actor_mean = build_mlp_net(arch=self._hp.actor_network_arch,
                                    input_size=self._actor_feature_extractor.encoding_size(),
                                    output_size=self._action_len,
                                    # use_weightnorm=True,
                                    use_torchscript=True,
                                    # weight_init_multiplier=0.01,
                                    # hidden_activations=th.nn.Tanh,
                                    layer_init_func=lambda m: ortho_layer_init_(m,1**0.5),
                                    last_layer_init_func=lambda m: ortho_layer_init_(m,0.01,action_mean_init)).to(device=self._hp.th_device)

        self.actor_logstd = nn.Parameter(th.full((1, self._action_len), self._hp.init_actor_logstd, device=self._hp.th_device))
        # Pre-compute the pre-tanh scale used to squash the actor mean to a stricter bound than ±1.
        # Output post-tanh mean lies in [-actor_mean_bounds_ratio, actor_mean_bounds_ratio].
        eps = 1e-6
        ratio = float(self._hp.actor_mean_bounds_ratio)
        if ratio <= 0.0 or ratio > 1.0:
            raise RuntimeError(f"actor_mean_bounds_ratio must be in (0, 1], got {ratio}")
        self._apply_mean_bounds = ratio < 1.0
        self._mean_bounds_pretanh_scale : float = math.atanh(min(ratio, 1.0 - eps))
        if self._hp.q_lr is not None and self._hp.q_lr!=self._hp.policy_lr:
            raise NotImplementedError("Different learning rates for Q and policy are not supported yet.")
        self._optimizer = AdamW(self.parameters(), lr=self._hp.policy_lr, eps=1e-8)
        self._grad_step_count = 0
        self._grad_step_count_th = th.as_tensor(0, device=self._hp.th_device)
        if self._hp.minibatch_size is None:
            if self._hp.minibatch_num is not None:
                self._hp.minibatch_size = self._hp.num_envs*self._hp.num_steps // self._hp.minibatch_num
            else:
                raise RuntimeError("Either minibatch_size or minibatch_num must be provided.")
        else:
            minibatch_num = self._hp.num_envs*self._hp.num_steps // self._hp.minibatch_size
            if self._hp.minibatch_num is not None and minibatch_num != self._hp.minibatch_num:
                raise RuntimeError("Provide only one of minibatch_size or minibatch_num, not both.")
            else:
                self._hp.minibatch_num = minibatch_num
        if self._hp.num_envs*self._hp.num_steps % self._hp.minibatch_size != 0:
            raise RuntimeError(f"num_envs*num_steps must be a multiple of minibatch_size, but num_envs={self._hp.num_envs}, num_steps={self._hp.num_steps} and minibatch_size={self._hp.minibatch_size}")
        self.__batch_size : Final[int] = int(self._hp.num_envs*self._hp.num_steps)
        self.__minibatch_num : Final[int] = int(self.__batch_size/self._hp.minibatch_size)


        self._policy_losses_sum = th.zeros((), device=self._hp.th_device)
        self._value_losses_sum = th.zeros((), device=self._hp.th_device)
        self._entropy_losses_sum = th.zeros((), device=self._hp.th_device)
        self._last_iter_policy_losses = th.zeros((self.__minibatch_num,), device=self._hp.th_device)
        self._last_iter_value_losses = th.zeros((self.__minibatch_num,), device=self._hp.th_device)
        self._last_iter_entropy_losses = th.zeros((self.__minibatch_num,), device=self._hp.th_device)
        self.__stats = {"tot_grad_steps_count":th.as_tensor(0, device=self._hp.th_device),
                        "q_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "actor_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "entropy_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        # "all_q_losses":[],
                        # "all_policy_losses":[],
                        # "all_entropy_losses":[],
                        "avg_q_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_actor_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_entropy_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_actor_logstd":th.as_tensor(float("nan"), device=self._hp.th_device),}
        self._log_full_loss_curve = False
        self._log_highest_q_loss_obss = False
        self._loss_table = wandb.Table(columns=["ppo_grad_step", "policy_loss", "value_loss", "entropy_loss", "loss"], log_mode="MUTABLE")


    @override
    def input_device(self):
        return self._hp.th_device

    @th.compile(fullgraph=True, mode="max-autotune")
    def get_value(self, obs_batch):
        critic_obs = self.get_critic_subobservation(obs_batch)
        enc_critic_obs_batch = self._critic_feature_extractor.extract_features(critic_obs)
        return self.critic(enc_critic_obs_batch)

    def get_actor_subobservation(self, observation):
        if self._hp.actor_observation_filter is None:
            return observation
        if not isinstance(observation, dict):
            raise RuntimeError(f"actor_observation_filter requires dict observations, got {type(observation)}")
        return {k: observation[k] for k in self._hp.actor_observation_filter}

    def get_critic_subobservation(self, observation):
        if self._hp.critic_observation_filter is None:
            return observation
        if not isinstance(observation, dict):
            raise RuntimeError(f"critic_observation_filter requires dict observations, got {type(observation)}")
        return {k: observation[k] for k in self._hp.critic_observation_filter}

    @staticmethod
    def _tanh_log_determinant(action):
        tanh_log_det_per_dim = 2.0 * (math.log(2.0) - action - th.nn.functional.softplus(-2.0 * action))
        tanh_log_determinant = tanh_log_det_per_dim.sum(1)
        return tanh_log_determinant

    @th.compile(fullgraph=True, mode="max-autotune")
    def get_action_and_extras(self, obs_batch=None, enc_actor_obs_batch = None, action=None):
        if enc_actor_obs_batch is None:
            actor_obs = self.get_actor_subobservation(obs_batch)
            enc_actor_obs_batch = self._actor_feature_extractor.extract_features(actor_obs)
        action_mean = self.actor_mean(enc_actor_obs_batch)
        # Squash the mean to a stricter bound than the post-tanh action bounds, so the tanh
        # squashing applied to action = mean + noise*std does not eat the noise asymmetrically
        # near ±1. The post-tanh mean ends up in [-actor_mean_bounds_ratio, actor_mean_bounds_ratio].
        # The log-prob correction for the tanh squashing is unaffected: this only reshapes the
        # pre-tanh mean, the Normal(mean, std) density and the tanh jacobian are unchanged.
        if self._apply_mean_bounds:
            scale = self._mean_bounds_pretanh_scale
            action_mean = th.tanh(action_mean / scale) * scale
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        had_input_action = action is not None
        action_sampled = action_mean + th.empty_like(action_mean).normal_(mean=0.0, std=1.0)*action_std # rsample has issues with torch.compile
        if not had_input_action:
            action = action_sampled

        # action is x_t (pre-tanh). The env receives tanh(x_t) in (-1,1).
        # Jacobian correction for tanh squashing: log|d tanh(x)/dx| = log(1-tanh(x)^2)
        # Numerically stable form (Brax TanhBijector): 2*(log2 - x - softplus(-2x))
        act_tanh_log_determinant = self._tanh_log_determinant(action)
        act_log_prob = probs.log_prob(action).sum(1) - act_tanh_log_determinant
        action_entropy = probs.entropy().sum(1) + act_tanh_log_determinant
        squashed_action = th.tanh(action)
        if had_input_action:
            sampled_act_entropy = probs.entropy().sum(1) + self._tanh_log_determinant(action_sampled)
        else:
            sampled_act_entropy = action_entropy
        # Brax-style Monte Carlo estimate of the squashed-policy entropy:
        # H[tanh(X)] = H[X] + E[log |d tanh(X)/dX|], approximated with the sampled action.
        return action, squashed_action, act_log_prob, action_entropy, action_mean, sampled_act_entropy


    @th.compile(fullgraph=True, mode="max-autotune")
    def get_action_logprob_entropy_critic_mean(self, obs_batch=None, enc_actor_obs_batch = None, enc_critic_obs_batch = None, action=None):
        action, squashed_action, act_log_prob, action_entropy, action_mean, sampled_act_entropy = self.get_action_and_extras(obs_batch, enc_actor_obs_batch, action)
        if enc_critic_obs_batch is None:
            if self._share_actor_critic_feature_extractor and enc_actor_obs_batch is not None:
                enc_critic_obs_batch = enc_actor_obs_batch
            else:
                critic_obs = self.get_critic_subobservation(obs_batch)
                enc_critic_obs_batch = self._critic_feature_extractor.extract_features(critic_obs)
        return action, squashed_action, act_log_prob, action_entropy, self.critic(enc_critic_obs_batch), action_mean, sampled_act_entropy

    @th.compile(fullgraph=True, mode="max-autotune")
    def _compute_losses(self, iteration, b_inds, b_actor_encobs, b_critic_encobs, b_actions, b_logprobs, b_values, b_advantages, b_returns) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:

        if self._hp.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # extract minibatch from batch
        start = iteration*self._hp.minibatch_size
        end = start + self._hp.minibatch_size
        mb_inds = b_inds[start:end]
        mb_actor_encobs = b_actor_encobs[mb_inds]
        mb_critic_encobs = b_critic_encobs[mb_inds]
        mb_acts = b_actions[mb_inds]
        mb_logprobs = b_logprobs[mb_inds]
        mb_values = b_values[mb_inds]
        mb_advantages = b_advantages[mb_inds]
        mb_returns = b_returns[mb_inds]

        _, _, newlogprobs, entropies, newvalues, _, sampled_action_entropies = self.get_action_logprob_entropy_critic_mean(  obs_batch=None,
                                                                                                enc_actor_obs_batch=mb_actor_encobs,
                                                                                                enc_critic_obs_batch=mb_critic_encobs,
                                                                                                action=mb_acts)
        logratio = newlogprobs - mb_logprobs
        ratio = logratio.exp()

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self._hp.epsilon_policy_ratio_clip, 1 + self._hp.epsilon_policy_ratio_clip)
        policy_loss = th.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = newvalues.view(-1)
        if self._hp.clip_vloss:
            v_loss_unclipped = (newvalues - mb_returns) ** 2
            v_clipped = mb_values + th.clamp(
                newvalues - mb_values,
                -self._hp.epsilon_value_clip_epsilon,
                self._hp.epsilon_value_clip_epsilon,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
            vec_value_loss = 0.5 * v_loss_max
        else:
            vec_value_loss = 0.5 * ((newvalues - mb_returns) ** 2)
        value_loss = vec_value_loss.mean()
        if not self._log_highest_q_loss_obss:
            vec_value_loss = None

        entropy_loss = - sampled_action_entropies.mean()
        loss = policy_loss + self._hp.loss_entropy_weight * entropy_loss + value_loss * self._hp.loss_value_weight

        with th.no_grad():
            self._policy_losses_sum += policy_loss
            self._value_losses_sum += value_loss
            self._entropy_losses_sum += entropy_loss
        self._last_iter_policy_losses[iteration] = policy_loss.detach()
        self._last_iter_value_losses[iteration] = value_loss.detach()
        self._last_iter_entropy_losses[iteration] = entropy_loss.detach()
        return loss, policy_loss, value_loss, entropy_loss, vec_value_loss

    def _compute_encoded_obss(self, raw_obss, num_steps, num_envs):
        raw_obss = map_tensor_tree(raw_obss, lambda l: l.flatten(0,1)) # flattn the first two dimensions (num_steps+1, num_envs)
        actor_obss = self.get_actor_subobservation(raw_obss)
        critic_obss = self.get_critic_subobservation(raw_obss)
        enc_actor_obss = self._actor_feature_extractor.extract_features(actor_obss)
        if self._share_actor_critic_feature_extractor:
            enc_critic_obss = enc_actor_obss
        else:
            enc_critic_obss = self._critic_feature_extractor.extract_features(critic_obss)
        enc_actor_obss = map_tensor_tree(enc_actor_obss, lambda l: l.view((num_steps+1, num_envs)+l.shape[1:]))
        enc_critic_obss = map_tensor_tree(enc_critic_obss, lambda l: l.view((num_steps+1, num_envs)+l.shape[1:]))
        return enc_actor_obss, enc_critic_obss

    def _compute_returns_and_advantages(self, rewards, terminateds, truncateds, values, consequent_values):
        advantages = th.zeros_like(rewards).to(self._hp.th_device)
        lastgaelam = 0
        for t in reversed(range(self._hp.num_steps)):
            nextnonterminal = 1.0 - terminateds[t + 1]
            nextnottruncated = 1.0 - truncateds[t + 1]
            nextnondone = nextnonterminal * nextnottruncated
            nextvalues = consequent_values[t]
            delta = rewards[t] + self._hp.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self._hp.gamma * self._hp.gae_lambda * nextnondone * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values[:-1]
        return returns, advantages

    def _reshape_batch_data(self,   enc_actor_obss : TensorTree[th.Tensor],
                                    enc_critic_obss : TensorTree[th.Tensor],
                                    actions : th.Tensor,
                                    logprobs : th.Tensor,
                                    values : th.Tensor,
                                    advantages : th.Tensor,
                                    returns : th.Tensor) -> tuple[TensorTree[th.Tensor], TensorTree[th.Tensor], th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        b_actor_encobs  = enc_actor_obss[:self._hp.num_steps].view((self.__batch_size,) + enc_actor_obss.size()[2:])
        b_critic_encobs = enc_critic_obss[:self._hp.num_steps].view((self.__batch_size,) + enc_critic_obss.size()[2:])
        b_logprobs      = logprobs[:self._hp.num_steps].view(self.__batch_size)
        b_actions       = actions.view((self.__batch_size,self._action_len))
        b_advantages    = advantages.view(self.__batch_size)
        b_returns       = returns.view(self.__batch_size)
        b_values        = values[:self._hp.num_steps].view(self.__batch_size)
        b_inds          = th.zeros(size=(self.__batch_size,), dtype=th.long, device=self._hp.th_device)
        return b_actor_encobs, b_critic_encobs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_inds

    def _prepare_epochs_data(self, raw_obss, actions, rewards, terminateds, truncateds, logprobs, start_values, num_steps, num_envs, consequent_values):
        # Extract obs features
        enc_actor_obss, enc_critic_obss = self._compute_encoded_obss(raw_obss, num_steps, num_envs)
        # bootstrap values and advantages
        with th.no_grad():
            returns, advantages = self._compute_returns_and_advantages(rewards, terminateds, truncateds, start_values, consequent_values)
        # flatten the batch
        return self._reshape_batch_data(enc_actor_obss, enc_critic_obss, actions, logprobs, start_values, advantages, returns)

    @th.compile(fullgraph=True, mode="max-autotune")
    def _opt_step(self):
        simplified_clip_grad_norm_(list(self.parameters()), self._hp.max_grad_norm)
        self._optimizer.step()

    def _save_worst_observations(self, vec_value_loss, raw_obss, mb_inds, worst_num=5):
        with th.no_grad():
            folder = adarl.utils.session.default_session.log_folder()+"/ppo_logs"
            os.makedirs(folder, exist_ok=True)
            ggLog.info(f"Saving worst {worst_num} obs with value loss of size {vec_value_loss.shape}...")
            top_indices = th.argsort(vec_value_loss, dim=0, descending=True)[:worst_num]
            for i in range(worst_num):
                index : int = top_indices[i].item()
                # observations are trajectories of shape (num_steps+1, num_envs, ...)
                # The value losses, correspond to obs-action pairs in the trajectories, however, the last observation
                # in each trajectory has been dropped and then, they have been shuffled and cut in minibatches.
                #  we have (num_steps, num_envs) value_losses.
                # So to get the raw_obs for a specific loss, we have num_step=index//num_envs and env_index=index%numenvs
                # We want to save the whole trajectory, together with the loss trajectory
                b_index = mb_inds[index]
                env_index = b_index % self._hp.num_envs
                step_index = b_index // self._hp.num_envs
                obs_traj = {k:v[:-1, env_index] for k,v in raw_obss.items()} # we save the obs at t and t+1, to have the obs corresponding to the action that generated the loss, and the next obs to see if it's a terminal state or not
                # loss_traj = vec_value_loss.view((self._hp.num_steps, self._hp.num_envs))[:, env_index]
                bad_loss_mask = th.zeros((self._hp.num_steps,), device=vec_value_loss.device)
                bad_loss_mask[step_index] = 1.0
                vl = vec_value_loss[index].item()
                obs_labels = {k:v.labels for k,v in self._critic_observation_space.spaces.items()}
                data = {"obs_traj": obs_traj,
                        # "loss_traj": loss_traj,
                        "bad_loss_mask": bad_loss_mask}
                data = map_tensor_tree(data, lambda t: t.cpu().numpy())
                hdf5plot.save.save_dict(folder+f"/worst_obs_{self._grad_step_count}_{i}_{vl}.hdf5",
                                        data=data,
                                        labels={"obs_traj": obs_labels})

    @override
    def train_model(self, buff : PPORolloutBuffer):
        # t_0 = time.monotonic()
        raw_start_obss, actions, rewards, terminateds, truncateds, logprobs, start_values, consequent_values = buff.get_rollout_data()
        # prepare encoded observations, returns and advantages, and flatten the numenv and trajectory dimensions together
        (b_actor_encobs,
         b_critic_encobs,
         b_logprobs,
         b_actions,
         b_advantages,
         b_returns,
         b_values,
         b_inds) = self._prepare_epochs_data(raw_start_obss, actions, rewards, terminateds, truncateds, logprobs, start_values, buff.num_steps, buff.num_envs,
                                             consequent_values)
        # clipfracs = []
        # t_pretrain = time.monotonic()
        self._policy_losses_sum.fill_(0)
        self._value_losses_sum.fill_(0)
        self._entropy_losses_sum.fill_(0)
        self._last_iter_policy_losses.fill_(0)
        self._last_iter_value_losses.fill_(0)
        self._last_iter_entropy_losses.fill_(0)
        for epoch in range(self._hp.update_epochs):
            th.randperm(self.__batch_size, out=b_inds, device=self._hp.th_device)
            # t_it0 = time.monotonic()
            for i in range(self.__minibatch_num):
                th.compiler.cudagraph_mark_step_begin()

                # with th.no_grad():
                #     old_approx_kl = (-logratio).mean()
                #     approx_kl = ((ratio - 1) - logratio).mean()
                #     clipfracs += [((ratio - 1.0).abs() > self._hp.clip_coef).float().mean().item()]

                loss, policy_loss, value_loss, entropy_loss, vec_value_losses = self._compute_losses( i,
                                                                                    b_inds,
                                                                                    b_actor_encobs,
                                                                                    b_critic_encobs,
                                                                                    b_actions,
                                                                                    b_logprobs,
                                                                                    b_values,
                                                                                    b_advantages,
                                                                                    b_returns)
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self._log_full_loss_curve:
                    self._loss_table.add_data(self._grad_step_count, policy_loss.detach().clone(), value_loss.detach().clone(), entropy_loss.detach().clone(), loss.detach().clone())
                with th.no_grad():
                    self._opt_step()
                if self._log_highest_q_loss_obss:
                    if self._grad_step_count % 1000 == 10:
                        start = i*self._hp.minibatch_size
                        end = start + self._hp.minibatch_size
                        mb_inds = b_inds[start:end]
                        self._save_worst_observations(vec_value_losses, raw_start_obss, mb_inds)
                self._grad_step_count_th += 1
                self._grad_step_count += 1
            # if self._hp.target_kl is not None and approx_kl > self._hp.target_kl:
            #     break
        if self._log_full_loss_curve:
            wandb.log({"ppo_losses": wandb.plot.line(self._loss_table, "ppo_grad_step", "loss", title="Custom Y vs X Line Plot 2")})
        tot_iterations = self._hp.update_epochs * self.__minibatch_num
        self.__stats.update({"tot_grad_steps_count":self._grad_step_count_th,
                            "q_loss":value_loss,
                            "actor_loss":policy_loss,
                            "entropy_loss":entropy_loss,
                            # "all_q_losses":self._last_iter_value_losses,
                            # "all_policy_losses":self._last_iter_policy_losses,
                            # "all_entropy_losses":self._last_iter_entropy_losses,
                            "avg_q_loss":self._value_losses_sum/tot_iterations,
                            "avg_actor_loss":self._policy_losses_sum/tot_iterations,
                            "avg_entropy_loss":self._entropy_losses_sum/tot_iterations,
                            "avg_actor_logstd":self.actor_logstd.mean()})
        # t_f = time.monotonic()
        # ggLog.info(f"took {t_f-t_0}, train={t_f-t_pretrain}, adv={t_postadv-t_postenc}, its={tit/self._hp.update_epochs}, samp={tit_sample} net={tit_net} comp={tit_comp} back={tit_back} op={tit_op}")
        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    def get_stats(self):
        return self.__stats

    @override
    def get_hidden_state(self):
        return None

    @override
    def reset_hidden_state(self):
        pass

    @override
    def predict_action(self, observation_batch, deterministic = False, extra_returns : dict | None = None):
        act, squashed_act, logprob, entropy, mean, _ = self.get_action_and_extras(obs_batch=observation_batch)
        if deterministic:
            return th.tanh(mean).detach().clone()
        else:
            return squashed_act.detach().clone()

    @override
    def save(self, path : str):
        with zipfile.ZipFile(path, mode="w") as archive:
            with archive.open("ppo.pth", "w") as ppo_file:
                th.save(self.state_dict(), ppo_file)
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
                extra_file.write(yaml.dump(extra, default_flow_style=None).encode("utf-8"))
            self._critic_feature_extractor.save_to_archive(archive, name="critic_feature_extractor")
            self._actor_feature_extractor.save_to_archive(archive, name="actor_feature_extractor")

    def _check_feature_extractor(self, current_featur_extractor : FeatureExtractor, loaded_fe_name, loaded_fe_args):
        if current_featur_extractor.__class__.__name__ != loaded_fe_name:
            ggLog.warn(f"feature_extractor_class_name of loaded model differs from that of self.\n"
                       f"loaded = {loaded_fe_name}, self's = {current_featur_extractor.__class__.__name__}")
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
            ggLog.warn(f"self init_args = \n{self_init_args_yaml}\n"
                       f"load init_args = \n{load_init_args_yaml}\n"
                       f"init_args diff = \n{diff}"
                       f"init args of loaded model differ from those of self.\n")
            # raise RuntimeError("Unmatched init_args")

    @override
    def load_(self, path : str):
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != self.__class__.__name__:
            raise RuntimeError(f"File was not saved by this class")
        curr_init_hparams = dataclasses.asdict(self._init_args["hyperparams"])
        load_init_hparams = dataclasses.asdict(extra["init_args"]["hyperparams"])
        equal, reasons = compare_dicts(curr_init_hparams, load_init_hparams)
        # equal, reasons = compare_dicts(self._init_args, extra["init_args"])
        if not equal:
            ggLog.warn("init args of loaded model differ from those of self.")
            load_yaml_args = yaml.dump(load_init_hparams)
            original_yaml_args = yaml.dump(curr_init_hparams)
            ggLog.warn(f"self._init_args = \n{original_yaml_args}")
            ggLog.warn(f"load init_args  = \n{load_yaml_args}")
            ggLog.warn(f"Differing fields: \n{reasons}")

        self._check_feature_extractor(self._critic_feature_extractor,
                                      extra["critic_feature_extractor_class_name"],
                                      extra["critic_feature_extractor_init_args"])
        self._check_feature_extractor(self._actor_feature_extractor,
                                      extra["actor_feature_extractor_class_name"],
                                      extra["actor_feature_extractor_init_args"])
        with zipfile.ZipFile(path) as archive:
            with archive.open("ppo.pth", "r") as ppo_file:
                state_dict = th.load(ppo_file)
                stripped = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
                model_keys = set(self.state_dict().keys())
                if any("._orig_mod." in k for k in model_keys):
                    stripped_to_model = {k.replace("._orig_mod.", "."): k for k in model_keys}
                    state_dict = {stripped_to_model.get(k, k): v for k, v in stripped.items()}
                else:
                    state_dict = stripped
                self.load_state_dict(state_dict)

    @classmethod
    def load(cls,   path : str, 
                    device : th.device | None = None,
                    init_args_override : dict | None = None):
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class")
        ppo_init_args = extra["init_args"]
        if device is not None:
            ppo_init_args["hyperparams"].th_device = device
        with zipfile.ZipFile(path) as archive:
            critic_feature_extractor_class = get_feature_extractor(extra["critic_feature_extractor_class_name"])
            ppo_init_args["critic_feature_extractor"] = critic_feature_extractor_class.load(archive, name="critic_feature_extractor", device=device)
            if extra["share_feature_extractor"]:
                ppo_init_args["actor_feature_extractor"] = ppo_init_args["critic_feature_extractor"]
            else:
                actor_feature_extractor_class = get_feature_extractor(extra["actor_feature_extractor_class_name"])
                ppo_init_args["actor_feature_extractor"] = actor_feature_extractor_class.load(archive, name="actor_feature_extractor", device=device)
        override_struct(ppo_init_args, init_args_override)
        ggLog.info(f"PPO.load(): building model with args: \n"+pprint.pformat(ppo_init_args))
        model = cls(**ppo_init_args)
        # At this point we should have a model that is initialized exactly like the one that was saved
        # So we can load into it the state from the checkpoint
        model.load_(path)
        return model
    
    @override
    def get_reference_init_args(self) -> dict:
        return self._hp.reference_init_args

register_agent_class(PPO)


class Collector():
    def __init__(self, vec_env_builder : VecEnvBuilderProtocol,
                   th_device : th.device,
                    env_builder_args : dict[str,Any],
                    run_folder : str,
                    seed : int,
                    num_envs : int):
        self._vec_env = vec_env_builder(seed=seed,
                                        run_folder=run_folder,
                                        num_envs=num_envs,
                                        env_builder_args=env_builder_args)
        action_space = self._vec_env.unwrapped.single_action_space
        obs_space = self._vec_env.unwrapped.single_observation_space
        self._num_envs = self._vec_env.unwrapped.num_envs

        if not isinstance(action_space, (spaces.gym_spaces.Box, spaces.gym_spaces.box.Box)):
            raise NotImplementedError(f"unsupported action space {action_space} of type {type(action_space)}")
        self._single_action_space = action_space
        # if not isinstance(obs_space, spaces.gym_spaces.Dict):
        #     raise NotImplementedError(f"unsupported observation space {obs_space}")
        self._single_observation_space = obs_space

        self._latest_start_obs, _ = self._vec_env.reset(seed=seed)
        # self._latest_obs = th.Tensor(self._latest_obs).to(th_device)
        self._latest_terminated = th.zeros(self._num_envs).to(th_device)
        self._latest_truncated = th.zeros(self._num_envs).to(th_device)
        self._device = th_device
        self._vec_env = self._vec_env
        self._env_device = th.device("cuda")
        self._policy_device = th.device("cpu")

    def single_observation_space(self):
        return self._single_observation_space

    def single_action_space(self):
        return self._single_action_space

    def num_envs(self):
        return self._num_envs

    def collect(self, vsteps_to_collect : int, buffer : PPORolloutBuffer, agent : PPO):
        term_count = th.as_tensor(0).to(device=self._env_device, non_blocking=True)
        with th.no_grad():
            buffer.reset()
            step_start_obs = map_tensor_tree(self._latest_start_obs, lambda a: th.as_tensor(a, device = agent.input_device()))
            prev_terminated = self._latest_terminated
            prev_truncated = self._latest_truncated
            for step in range(0, vsteps_to_collect):
                # print(f"collecting step {step}/{vsteps_to_collect}")
                th.compiler.cudagraph_mark_step_begin()
                # ALGO LOGIC: action logic
                raw_action, squashed_action, logprob, _, start_value, _, _ = agent.get_action_logprob_entropy_critic_mean(obs_batch=step_start_obs)
                start_value = start_value.clone()
                raw_action = raw_action.clone()  # pre-tanh x_t, stored in buffer for log_prob recomputation
                squashed_action = squashed_action.clone().to(device=self._env_device)  # tanh(x_t) in (-1,1), sent to env
                logprob = logprob.clone()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_start_obs, reward, terminations, truncations, info = self._vec_env.step(squashed_action)
                next_start_obs = map_tensor_tree(next_start_obs, lambda a: th.as_tensor(a))
                consequent_obs = info.get("final_observation",next_start_obs)
                consequent_obs = map_tensor_tree(consequent_obs, lambda a: th.as_tensor(a))
                # The consequent value computation may be optimized by moving it inside the next step get_action_logprob_entropy_critic_mean
                consequent_value = agent.get_value(map_tensor_tree(consequent_obs, lambda a: th.as_tensor(a, device=agent.input_device()))).flatten()
                reward = th.as_tensor(reward)
                terminations = th.as_tensor(terminations)
                truncations = th.as_tensor(truncations)
                buffer.add( start_obss=step_start_obs,
                            start_obss_values=start_value.flatten(),
                            prev_terminateds=prev_terminated,
                            prev_truncateds=prev_truncated,
                            actions=raw_action,
                            logprobs=logprob,
                            rewards=reward.view(-1),
                            consequent_obss_values=consequent_value)
                step_start_obs = map_tensor_tree(next_start_obs, lambda t: t.detach().clone().to(device=agent.input_device(), non_blocking=t.device.type=="cuda"))
                done = th.logical_or(terminations, truncations)
                term_count += th.count_nonzero(done)
                # buffer._obs[step] = start_obs
                prev_terminated = terminations
                prev_truncated = truncations

            self._latest_terminated = terminations
            self._latest_truncated = truncations
            self._latest_start_obs = next_start_obs

            th.compiler.cudagraph_mark_step_begin()

            self._latest_start_obs = map_tensor_tree(self._latest_start_obs, lambda a: th.as_tensor(a, device = agent.input_device()))
            buffer.set( prev_terminateds=self._latest_terminated,
                        prev_truncateds=self._latest_truncated,
                        consequent_obss_values=consequent_value)
        return term_count








def train_on_policy(collector : Collector,
                    model : PPO,
                    num_steps : int,
                    storage_torch_device : th.device,
                    train_steps : int,
                    log_freq_vstep : int = -1,
                    callbacks : list[TrainingCallback] = []):

    buffer = PPORolloutBuffer(num_steps, collector.num_envs(),
                                obs_space=collector.single_observation_space(),
                                act_space_shape=collector.single_action_space().shape,
                                device=storage_torch_device)

    # buffer = PPORolloutBuffer(num_envs=collector.num_envs(),
    #                           num_steps=num_steps,
    #                           storage_torch_device=storage_torch_device,
    #                           observation_space=collector.observation_space(),
    #                           action_size=np.prod(collector.single_action_space().shape))
    callback = CallbackList(callbacks)
    callback.on_training_start()
    global_step = 0
    ep_counter = 0
    t_coll_sl = 0
    t_train_sl = 0
    t_tot_sl = 0
    steps_sl = 0
    last_log_steps = float("-inf")
    start_time = time.monotonic()
    while global_step < train_steps and not adarl.utils.session.default_session.is_shutting_down():
        with th.no_grad():
            callback.on_collection_start()
            t0 = time.monotonic()
            terminated_eps = collector.collect(buffer=buffer, vsteps_to_collect=num_steps, agent=model)
            t_post_coll = time.monotonic()
            t_coll_sl += t_post_coll-t0
            global_step += num_steps*collector.num_envs()
            steps_sl += num_steps*collector.num_envs()
            ep_counter += terminated_eps
            adarl.utils.session.default_session.run_info["collected_episodes"].value = ep_counter
            adarl.utils.session.default_session.run_info["collected_steps"].value = global_step
            callback.on_collection_end(collected_episodes=int(terminated_eps.item()),
                                    collected_steps=num_steps,
                                    collected_data=None)
            t_post_cb = time.monotonic()
        # import torch._dynamo as dynamo
        # explanation = dynamo.explain(model.train_model)(buffer)
        # print(explanation)
        # input("press ENTER")
        model.train_model(buffer)
        adarl.utils.session.default_session.run_info["train_iterations"].value = model._grad_step_count
        t_f = time.monotonic()
        t_train_sl += t_f-t_post_cb
        t_tot_sl += t_f - t0

        wlogs = {"ppo/"+k:v for k,v in model.get_stats().items()}
        wandb_log(wlogs,throttle_period=2, silent_throttling=True)
        if global_step - last_log_steps > log_freq_vstep*collector.num_envs():
            last_log_steps = global_step
            ggLog.info(f"ONTRAIN: expstps:{global_step}"
                        f" trainstps={model._grad_step_count}"
                        #    f" exp_reuse={model._tot_grad_steps_count*batch_size/global_step:.2f}"
                        f" coll={t_coll_sl:.2f}s train={t_train_sl:.2f}s tot={t_tot_sl:.2f}"
                        f" fps={steps_sl/t_tot_sl:.2f} collfps={steps_sl/t_coll_sl:.2f}"
                        f" alltime_fps={global_step/(t_f-start_time):.2f} alltime_ips={model._grad_step_count/(t_f-start_time):.2f}")
            t_coll_sl = 0
            t_train_sl = 0
            t_tot_sl = 0
            steps_sl = 0
            t_tot_sl = 0
            # t = f"{time.time():.3f}"
            # jax.profiler.save_device_memory_profile(f"jax_memory_{t}.prof")
            # th.cuda.memory._dump_snapshot(f"jax_memory_{t}.pickle")
            # free, total = th.cuda.mem_get_info(th.device('cuda:0'))
            # mem_used_MB = (total - free) / 1024 ** 2
            # ggLog.info(f"{t}: cuda mem usage = {mem_used_MB}")
        adarl.utils.sigint_handler.haltOnSigintReceived()
    callback.on_training_end()








@dataclass
class PPO_init_hyperparams():
    actor_network_arch : tuple[int,...]
    critic_network_arch : tuple[int,...]
    epsilon_policy_ratio_clip : float
    epsilon_value_clip_epsilon : float
    gae_lambda : float
    gamma : float
    log_freq_vstep : int
    loss_entropy_coeff : float
    loss_value_weight : float
    max_grad_norm : float
    minibatch_size : int | None
    minibatch_num : int | None
    num_envs : int
    num_steps : int
    policy_lr : float
    q_lr : float
    th_device : th.device
    total_steps : int
    update_epochs : int
    init_actor_logstd : float
    actor_observation_filter : list[str] | None = None
    critic_observation_filter : list[str] | None = None
    actor_mean_bounds_ratio : float = 1.0
    reference_init_args : dict = dataclasses.field(default_factory=dict)

def ppo_train(  seed : int,
                folderName : str,
                run_id : str,
                args,
                env_builder : EnvBuilderProtocol | None,
                vec_env_builder : VecEnvBuilderProtocol | None,
                env_builder_args : dict,
                agent_hyperparams : PPO_init_hyperparams,
                max_episode_duration : int,
                validation_buffer_size : int,
                validation_holdout_ratio : float,
                validation_batch_size : int,
                eval_configurations : list[dict] = [],
                checkpoint_freq_vec_ep : int = 100,
                collector_device : th.device | None = None,
                debug_level : int = 2,
                no_wandb : bool = False,
                env_checker_max_obs_value : float = 255.0,
                env_checker_max_rew_value : float = 100.0):

    #     th.cuda.memory._record_memory_history(
    #        max_entries=100_000
    #    )

    run_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        debug=debug_level,
                                                        use_wandb=not no_wandb)
    validation_enabled = validation_buffer_size > 0 or validation_holdout_ratio > 0 or validation_batch_size > 0

    random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    if agent_hyperparams.th_device == "cuda": agent_hyperparams.th_device = th.device("cuda",0)
    device = th.device(agent_hyperparams.th_device)
    if collector_device is None:
        collector_device = device
    if vec_env_builder is None and env_builder is not None:
        vec_env_builder = lambda seed, run_folder, num_envs, env_builder_args, env_name="": build_vec_env(   env_builder=env_builder,
                                                                                                env_builder_args=env_builder_args,
                                                                                                log_folder=run_folder,
                                                                                                seed=seed,
                                                                                                num_envs=num_envs,
                                                                                                collector_device=collector_device,
                                                                                                env_action_device=collector_device)
    if vec_env_builder is None:
        raise RuntimeError(f"You must specify either vec_env_builder or env_builder")
    vec_env_builder = wrap_with_logger(vec_env_builder,
                                       max_obs_value=env_checker_max_obs_value,
                                       max_rew_value=env_checker_max_rew_value)

    collector = Collector(  vec_env_builder=vec_env_builder,
                            env_builder_args=env_builder_args,
                            th_device=agent_hyperparams.th_device,
                            run_folder=run_folder,
                            num_envs=agent_hyperparams.num_envs,
                            seed=seed)
    action_space = collector.single_action_space()
    obs_space = collector.single_observation_space()
    agent = PPO(PPO_Hyperparams(minibatch_size=agent_hyperparams.minibatch_size,
                                minibatch_num=agent_hyperparams.minibatch_num,
                                th_device=agent_hyperparams.th_device,
                                action_space=action_space,
                                observation_space=obs_space,
                                action_max=th.as_tensor(action_space.high, device=agent_hyperparams.th_device),
                                action_min=th.as_tensor(action_space.low, device=agent_hyperparams.th_device),
                                q_lr=agent_hyperparams.q_lr,
                                policy_lr=agent_hyperparams.policy_lr,
                                num_envs=agent_hyperparams.num_envs,
                                num_steps=agent_hyperparams.num_steps,
                                gamma=agent_hyperparams.gamma,
                                update_epochs=agent_hyperparams.update_epochs,
                                actor_network_arch=agent_hyperparams.actor_network_arch,
                                critic_network_arch=agent_hyperparams.critic_network_arch,
                                actor_observation_filter=agent_hyperparams.actor_observation_filter,
                                critic_observation_filter=agent_hyperparams.critic_observation_filter,
                                loss_value_weight=agent_hyperparams.loss_value_weight,
                                loss_entropy_weight=agent_hyperparams.loss_entropy_coeff,
                                epsilon_policy_ratio_clip=agent_hyperparams.epsilon_policy_ratio_clip,
                                epsilon_value_clip_epsilon=agent_hyperparams.epsilon_value_clip_epsilon,
                                clip_vloss=agent_hyperparams.epsilon_value_clip_epsilon < float("+inf"),
                                gae_lambda=agent_hyperparams.gae_lambda,
                                max_grad_norm=agent_hyperparams.max_grad_norm,
                                init_actor_logstd=agent_hyperparams.init_actor_logstd,
                                actor_mean_bounds_ratio=agent_hyperparams.actor_mean_bounds_ratio,
                                reference_init_args=agent_hyperparams.reference_init_args))
    ggLog.info(f"Compiling PPO model...")
    t0 = time.monotonic()
    agent = th.compile(agent, fullgraph=True, mode="max-autotune")
    t1 = time.monotonic()
    ggLog.info(f"Torch model compilation took {t1-t0:.3f}s")

    # torchexplorer.watch(model, backend="wandb")
    # wandb.watch((agent, agent._actor, agent._critic), log="all", log_freq=1000, log_graph=False)

    # compiled_model = th.compile(model)

    # rb = ThDictEpReplayBuffer(  buffer_size=hyperparams.buffer_size,
    #                             observation_space=observation_space,
    #                             action_space=action_space,
    #                             device=device,
    #                             storage_torch_device=device,
    #                             n_envs=hyperparams.parallel_envs,
    #                             max_episode_duration=max_episode_duration,
    #                             validation_buffer_size = validation_buffer_size,
    #                             validation_holdout_ratio = validation_holdout_ratio,
    #                             min_episode_duration = 0,
    #                             disable_validation_set = False,
    #                             fill_val_buffer_to_min_at_step = hyperparams.learning_starts,
    #                             val_buffer_min_size = validation_batch_size)

    # ggLog.info(f"Replay buffer occupies {rb.memory_size()/1024/1024:.2f}MB on {rb.storage_torch_device()}")

    start_time = time.time()
    callbacks = build_eval_callbacks(eval_configurations=eval_configurations,
                                     vec_env_builder=vec_env_builder,
                                     run_folder=run_folder,
                                     base_seed=seed,
                                     collector_device=collector_device,
                                     model = agent)

    callbacks.append(CheckpointCallbackRB(save_path=run_folder+"/checkpoints",
                                          model=agent,
                                          save_best=False,
                                          save_freq_ep=checkpoint_freq_vec_ep*agent_hyperparams.num_envs))

    train_on_policy(collector = collector,
                    model = agent,
                    num_steps = agent_hyperparams.num_steps,
                    storage_torch_device = device,
                    train_steps = agent_hyperparams.total_steps,
                    callbacks = callbacks,
                    log_freq_vstep = agent_hyperparams.log_freq_vstep)



def example():

    seed = 0
    run_id = str(int(time.monotonic()))
    env_builder_args={"env_name" : "HalfCheetah-v4",
                    "forward_reward_weight" : 1.0,
                    "ctrl_cost_weight" : 0.1,
                    "reset_noise_scale" : 0.1,
                    "exclude_current_positions_from_observation" : True,
                    "max_episode_steps" : 1000,
                    "quiet" : True,
                    "clip_action" : True,
                    "normalize_obs" : True,
                    "clip_obs" : True,
                    "normalize_reward" : True,
                    "clip_reward" : True,
                    "dict_obs" : False}
    ppo_train(  seed=seed,
                folderName="./cleanrl_ppo",
                run_id=run_id,
                args={"comment":"t"},
                env_builder=gym_builder,
                vec_env_builder=None,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_init_hyperparams(  minibatch_size=512,
                                                    th_device=th.device("cuda"),
                                                    actor_network_arch=(64,64),
                                                    critic_network_arch=(64,64),
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=10,
                                                    total_steps=1_000_000,
                                                    num_envs=8,
                                                    num_steps=2048,
                                                    gamma=0.99,
                                                    log_freq_vstep = 1000),
                max_episode_duration=1000,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0,
                checkpoint_freq_vec_ep=-1,
                collector_device=th.device("cpu"))






if __name__ == "__main__":
    example()

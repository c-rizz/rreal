from __future__ import annotations
from gc import freeze
from adarl.utils.buffers import ThDReplayBuffer, TransitionBatch, BaseBuffer, BaseValidatingBuffer
from rreal.utils.callbacks import TrainingCallback, CallbackList
from adarl.utils.tensor_trees import sizetree_from_space, map2_tensor_tree, flatten_tensor_tree, map_tensor_tree, filter_by_key_tensor_tree, map_tensor_tree_withkey
from adarl.utils.wandb_wrapper import wandb_log
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check_size
from adarl.utils.utils import get_func_input_args, th_compile_ext, override_struct
from dataclasses import dataclass, asdict
from rreal.algorithms.collectors import ExperienceCollector
from rreal.algorithms.rl_agent import RLAgent, register_agent_class
from rreal.feature_extractors import get_feature_extractor
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor, StackVectorsFeatureExtractorInitArgs
from rreal.utils.utils import build_mlp_net, scale_layer_weights, split_params_for_weight_decay, simplified_clip_grad_norm_, filter_dict_space, get_params_with_decay_mask, filter_dict
from rreal.utils.rnd_hyperparams import SAC_RND_reward_hyperparams
from typing import List, Union, Literal, Mapping, Callable
import rreal.utils.callbacks
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
from typing import Protocol
import numpy as np
import typing
import math
# from torch.optim import AdamW
from rreal.utils.FixedAdamW import AdamW
import os

th._dynamo.config.compiled_autograd = True
th._dynamo.config.allow_unspec_int_on_nn_module = True

# There seems to be a BIG overhead in entering end exiting compiled functions
# I tried to dig a bit in the torch/dynamo/inductor code to understand at the end what is being ran when
# calling a compiled function, but its deeeeeeeep
compile_mode="max-autotune" # reduce overhead doesn't seem to reduce overhead more than max-autotune
disable_compile = False
dynamic_compile = False
fullgraph = False

DictObs = dict[str, th.Tensor]
@dataclass
class DictTransitionBatch():
    observations : DictObs
    actions : th.Tensor
    next_observations : DictObs
    terminated : th.Tensor
    rewards : th.Tensor

class TransitionAugmentorFunction(Protocol):
        def __call__(self,
                     observations : DictObs,
                     actions : th.Tensor,
                     next_observations : DictObs,
                     rewards : th.Tensor,
                     terminateds : th.Tensor
                     ) -> tuple[DictObs, th.Tensor, DictObs, th.Tensor, th.Tensor]:
            ...

def compare_dicts(d1 : dict, d2 : dict) -> tuple[bool, str]:
    all_keys = set(d1.keys()).union(set(d2.keys()))
    diffs = ""
    equal = True
    for k in sorted(all_keys):
        e1 = d1.get(k,None)
        e2 = d2.get(k,None)
        d = th.as_tensor(e1 != e2)
        if th.any(d):
            equal = False
            diffs += f"{k}: {e1} != {e2}\n"
    return equal, diffs

def nop_func(arg1):
    pass

def safe_quantile(x : th.Tensor, q : float, dim : int = 0) -> th.Tensor:
    """Quantile keeps giving issues with torch compile, this function tries to go around the problem

    Parameters
    ----------
    x : th.Tensor
        The input tensor.
    q : float
        The quantile to compute, should be between 0 and 1.
    dim : int, optional
        The dimension along which to compute the quantile, by default 0
    Returns
    -------
    th.Tensor
        The computed quantile values.
    """
    return th.kthvalue(x, k=int(q * x.size(dim)), dim=dim).values

class RewardAugmentorProtocol(Protocol):
        def __call__(self,
                     transitions : DictTransitionBatch,
                     critic_enc_obss : th.Tensor,
                     critic_enc_next_obss : th.Tensor,
                     actor_next_enc_obss : th.Tensor) -> th.Tensor:
            ...

class PostUpdateHookProtocol(Protocol):
        """Protocol for a hook ran after the SAC model update. 
        The hook receives:
         * The transition batch the model was trained on
         * The encoded observations for the transitions (encoded actor observation may be None)
         * The losses computed during the update (q_loss, actor_loss, alpha_loss)"""
        
        def __call__(self,
                     transitions : DictTransitionBatch,
                     encoded_obss : tuple[th.Tensor | None, th.Tensor, th.Tensor, th.Tensor],
                     losses : tuple[th.Tensor, th.Tensor, th.Tensor]) -> dict[str,th.Tensor]:
            ...

@typing.runtime_checkable
class AnnealingFunction(Protocol):
        def __call__(self,  global_exp_step : int, train_iterations : int) -> float:
            ...
def get_constant_annealing(value : float) -> AnnealingFunction:
    """
    Returns a function that always returns the same value.
    This is used for the target entropy in SAC.
    """
    def constant_annealing(global_exp_step : int, train_iterations : int) -> float:
        return value
    return constant_annealing

def get_ramp_annealing(ramp_start_step : int, ramp_end_step : int, start_value : float, end_value : float) -> AnnealingFunction:
    """
    Returns a function that ramps from start_value to end_value between ramp_start_step and ramp_end_step.
    """
    def ramp_annealing(global_exp_step : int, train_iterations : int) -> float:
        if train_iterations < ramp_start_step:
            return start_value
        elif train_iterations > ramp_end_step:
            return end_value
        else:
            progress = (train_iterations - ramp_start_step) / (ramp_end_step - ramp_start_step)
            return start_value + progress * (end_value - start_value)
    return ramp_annealing

annealings :dict[str, typing.Callable[..., AnnealingFunction]] = {
    "constant": get_constant_annealing,
    "ramp": get_ramp_annealing
}

@dataclass
class SAC_init_hparams:
    q_network_arch : list[int]
    """The architecture of the Q network, a list of hidden layer sizes"""
    policy_arch : list[int]
    """The architecture of the policy network, a list of hidden layer sizes"""
    q_lr : float
    """The learning rate for the Q network"""
    policy_lr : float
    """The learning rate for the policy network"""
    model_th_device : str | th.device
    """The torch device where the model will be located"""
    gamma : th.Tensor | Mapping[str, float | th.Tensor] | float
    """The discount factor for the Q-learning algorithm"""
    target_tau : float
    """The target update factor for the soft update of the target network, the smaller it is the more delayed the target is. Usually 0.005."""
    buffer_size : int
    """The size of the replay buffer"""
    total_steps : int
    """The total number of steps to train the model for"""
    train_freq_vstep : int
    """The numer of vectorized experience collection steps (i.e. steps/parallel_envs) between training steps"""
    learning_starts : int
    """The number of steps to collect before starting training"""
    grad_steps : int
    """The number of gradient steps to take per training step"""
    batch_size : int
    """The batch size used for computing gradient updates"""
    parallel_envs : int
    """The number of parallel environments to use for experience collection"""
    log_freq_vstep : int
    """The frequency of logging, in number of vectorized steps (i.e. steps/parallel_envs)"""
    reference_init_args : dict
    """Additional arguments that will be saved together with the model, just for reference on how it was trained"""
    target_entropy_factor : float | None
    """The factor used to compute the target entropy as target_entropy_factor*action_size, by default it is -1.0"""
    actor_log_std_init : float
    """The initial value of the log standard deviation of the actor's policy, by default it is -3.0"""
    actor_observation_filter : list[str] | None = None
    """The list of observation keys to filter in the actor's policy, by default it is None (no filtering, all observation keys are used)"""
    critic_observation_filter : list[str] | None = None
    """The list of observation keys to filter in the critic's Q network, by default it is None (no filtering, all observation keys are used)"""
    target_entropy_factor_annealing : tuple[Literal['constant', 'ramp'], list[float | th.Tensor]] | AnnealingFunction | None = None
    """The target entropy factor annealing function, by default it is None (no annealing), see predefined annealings in sac.py"""
    action_reference_obs_key : str | None = None
    """The observation key that will be used as a reference for the action, meaning the actor distribution is computed as `mean = NN(obs) + act_ref` 
      by default it is None (no reference, the mean is produced from the network directly)"""
    actor_mean_bounds_ratio : float = 1.0
    """The ratio of the action bounds that the actor's mean can reach, by default it is 1.0 (the mean can reach the action bounds). Reducing this can prevent boundary effects that reduce noise on the edges, biasing the actor toward them."""
    max_grad_norm : float = 0.5
    log_alpha_grad_clip : float = 0.1
    feature_extractor_lr : float = 0.001
    policy_update_freq : int = 2
    target_update_freq : int = 1
    auto_entropy_temperature : bool =True
    constant_entropy_temperature : float | None =None
    critic_weight_decay : float = 0.0
    actor_weight_decay : float = 0.0
    deterministic_collection_ratio : float = 0.0
    alpha_lr_factor : float = 1.0
    alpha_initial_value : float = 0.1
    independent_entropy_q : bool = False
    rnd_hyperparams : SAC_RND_reward_hyperparams | None = None
    """RND exploration configuration. When it is set and it asks for a separate reward channel,
    SAC appends a novelty channel to its reward vector, so that the bonus gets its own q value and
    its own discount instead of being added onto the task rewards. This must be part of the
    hyperparams, rather than applied to the built agent, because the collectors build their own
    inference copies from the very same hyperparams and their state_dicts have to match.
    The novelty estimator itself is still attached separately, so that only the trained agent
    carries one."""


def _append_reward_channel(reward_space : spaces.ThBox, name : str, torch_device) -> spaces.ThBox:
    """Returns a copy of reward_space with one extra unbounded channel appended.

    The channel is unbounded on both sides because the novelty bonus is signed: samples that are
    less novel than average get a negative one. QNetwork derives its output bounds from the signs
    of the reward space limits, so finite one-sided limits here would clip the novelty q value.
    """
    low = np.concatenate([np.asarray(reward_space.low).reshape(-1), [-np.inf]])
    high = np.concatenate([np.asarray(reward_space.high).reshape(-1), [np.inf]])
    labels = getattr(reward_space, "labels", None)
    if labels is None:
        labels = np.array([f"reward_r{i:03d}" for i in range(low.shape[0]-1)], dtype=object)
    labels = np.concatenate([np.asarray(labels, dtype=object).reshape(-1),
                             np.array([name], dtype=object)])
    return spaces.ThBox(low=low, high=high, dtype=np.float32,
                        torch_device=th.device(torch_device), labels=labels)


class SignedELUBounding(nn.Module):
    """Bounds vector entries to positive-only or negative-only ranges using ELU."""

    def __init__(self, q_bounds_minmax: th.Tensor, x_offset : th.Tensor | float = -4.0):
        """ Initialize the bounding layer

        Parameters
        ----------
        q_bounds_minmax : th.Tensor
            A tensor of shape [2, N] containing the min and max bounds for each of the N components of the input vector.
        x_offset : th.Tensor | float, optional
            The offset for the ELU activation function. Useful to bring the activation closer to an identity. With
            x_offset=0 SignedELUBounding(0) = ±1, with x_offset=-4 SignedELUBounding(-4) ≈ ±0.0183.
            by default -4.0

        Raises
        ------
        NotImplementedError
            _description_
        """
        super().__init__()
        if q_bounds_minmax.dim() != 2 or q_bounds_minmax.size(0) != 2:
            raise NotImplementedError("SignedELUBounding expects bounds shaped [2, N]")
        self._vector_size = q_bounds_minmax.shape[1]
        self._x_offset = x_offset
        self._bounded_positive_mask = q_bounds_minmax[0] >= 0
        self._bounded_negative_mask = q_bounds_minmax[1] <= 0
        self._needs_positive_bounding = bool(th.any(self._bounded_positive_mask).item())
        self._needs_negative_bounding = bool(th.any(self._bounded_negative_mask).item())
        self._needs_bounding = self._needs_positive_bounding or self._needs_negative_bounding

    def forward(self, q_values: th.Tensor) -> th.Tensor:
        # if q_values.size(-1) != self._vector_size:
        #     raise NotImplementedError("SignedELUBounding expects last dimension to match bounds size")
        if not self._needs_bounding:
            return q_values
        
        if self._needs_positive_bounding:
            bounded_positive = F.elu(q_values+self._x_offset) + 1.0
            q_values = th.where(self._bounded_positive_mask, bounded_positive, q_values)
        if self._needs_negative_bounding:
            bounded_negative = - (F.elu(-q_values+self._x_offset) + 1.0)
            q_values = th.where(self._bounded_negative_mask, bounded_negative, q_values)
        return q_values

class QNetwork(nn.Module):
    def __init__(self,
                 action_size : int,
                 q_network_arch : List[int],
                 observation_size : int,
                 torch_device : Union[str,th.device] = "cuda",
                 nets_num : int = 1,
                 initial_scale = 0.003,
                 use_weightnorm : bool = True,
                 reward_space : spaces.ThBox | None = None,
                 inner_activations : Callable[[],nn.Module] = nn.Tanh,
                 independent_entropy_q : bool = False):
        super().__init__()
        self._nets_num = nets_num
        self._obs_size = observation_size
        self._use_weightnorm = use_weightnorm
        self._independent_entropy_q = independent_entropy_q
        if reward_space is not None:
            rewards_num = spaces.get_1d_space_size(reward_space) if reward_space is not None else 1
            has_negative_rewards = th.as_tensor(reward_space.low, device=torch_device) < 0
            has_positive_rewards = th.as_tensor(reward_space.high, device=torch_device) > 0
            ones = th.ones((rewards_num,), device=torch_device, dtype=th.float32)
            zeros = th.zeros((rewards_num,), device=torch_device, dtype=th.float32)
            q_bounds_minmax = th.stack([th.where(has_negative_rewards, float("-inf")*ones, zeros),
                                        th.where(has_positive_rewards, float("+inf")*ones, zeros)],
                                dim=0).view(2,rewards_num)
        else:
            rewards_num = 1
            q_bounds_minmax = th.as_tensor([float("-inf"), float("+inf")]).view(2,1)
        if self._independent_entropy_q:
            q_bounds_minmax = th.cat([q_bounds_minmax,
                                      th.as_tensor([[float("-inf")],[float("+inf")]], device=torch_device)],
                                     dim=1)
            self._q_size = rewards_num + 1
        else:
            self._q_size = rewards_num
        self._q_nets = build_mlp_net(arch=q_network_arch,
                                     input_size=action_size + observation_size,
                                     output_size=self._q_size,
                                     ensemble_size=self._nets_num,
                                     return_ensemble_mean=False,
                                     use_weightnorm=self._use_weightnorm,
                                     use_torchscript=False,
                                     use_jit_fork=False,
                                     hidden_activations=inner_activations,
                                     last_layer_init_func= lambda m: scale_layer_weights(m,initial_scale)).to(device=torch_device)
        self._bounding_layer = SignedELUBounding(q_bounds_minmax).to(device=torch_device)
    
    # @th.compile(mode=compile_mode, fullgraph=fullgraph)
    def get_min_qval(self, observations, actions):
        qvals = self(observations, actions)
        # ggLog.info(f"qvals.size() = {qvals.size()}")
        # min_q = qvals[:,0]
        min_q = th.amin(qvals,dim=1)
        # min_q = min_q.squeeze(1)
        # ggLog.info(f"min_q.size() = {min_q.size()}")
        return min_q.view(-1, self._q_size)
    
    # @th.compile(mode=compile_mode, fullgraph=fullgraph)    
    def forward(self, observations, actions):
        qvals = self._q_nets(th.cat([observations, actions], 1))
        qvals = qvals.view(-1, self._nets_num, self._q_size)
        qvals = self._bounding_layer(qvals)
        # ggLog.info(f"QNetwork.forward: qvals.size() = {qvals.size()}")
        return qvals



class Actor(nn.Module):
    def __init__(self,  action_size,
                        observation_size : int,
                        policy_arch = [256,256],
                        action_max : Union[float, th.Tensor] = 1,
                        action_min : Union[float, th.Tensor] = -1,
                        log_std_max = 2,
                        log_std_min = -8,
                        log_std_init = -3.0,
                        init_noise = 0.001,
                        torch_device : Union[str,th.device] = "cuda",
                        action_mean_init : float | th.Tensor= 0.0,
                        use_weightnorm : bool = True,
                        mean_bounds_ratio : float | None = None,
                        inner_activations : Callable[[],nn.Module] = th.nn.Tanh,
                        dtype : th.dtype = th.float32):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self.device = torch_device
        self.dtype = dtype
        self._obs_size = observation_size
        self._use_weightnorm = use_weightnorm
        self._mean_bounds_ratio = mean_bounds_ratio if mean_bounds_ratio is not None else 1.0
        self._action_size = action_size
        # ggLog.info(f"Actor mean_bounds_ratio = {self._mean_bounds_ratio}")
        # ggLog.info(f"Actor action_max = {action_max} action_min = {action_min}")

        # save action scaling factors as non-trained parameters
        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=self.dtype)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=self.dtype)
        action_max = action_max.to(device=torch_device)
        action_min = action_min.to(device=torch_device)
        self.action_bias : th.Tensor
        self.action_scale : th.Tensor
        self.register_buffer("action_scale", th.as_tensor((action_max - action_min) / 2.0, dtype=self.dtype, device=torch_device))
        self.register_buffer("action_bias",  th.as_tensor((action_max + action_min) / 2.0, dtype=self.dtype, device=torch_device))
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.act_fc = build_mlp_net(arch=policy_arch[:-1],input_size=observation_size, output_size=policy_arch[-1],
                                    last_activation_class=inner_activations,
                                    hidden_activations=inner_activations,
                                    use_weightnorm=self._use_weightnorm).to(device=torch_device)
        action_mean_init = th.as_tensor(action_mean_init, dtype=self.dtype).to(device=torch_device)
        if th.any((action_mean_init > action_max) | (action_mean_init < action_min)):
            ggLog.warn( f"action_mean_init has values outside bounds:\n"
                        f"action_mean_init : {action_mean_init}\n"
                        f"action_max : {action_max}\n"
                        f"action_min : {action_min}\n")
            action_mean_init = th.clamp(action_mean_init, min=action_min*0.99, max=action_max*0.99)
        mean_init = th.atanh((action_mean_init-self.action_bias)/self.action_scale).to("cpu")
        self.act_fc_mean = build_mlp_net(arch=[],
                                         input_size=policy_arch[-1],
                                         output_size=action_size,
                                         use_weightnorm=self._use_weightnorm,
                                         last_layer_init_func=lambda m: scale_layer_weights(m, init_noise, bias_offset=mean_init),
                                         hidden_activations=inner_activations).to(device=torch_device,)
        self.act_fc_logstd = build_mlp_net(arch=[],
                                         input_size=policy_arch[-1],
                                         output_size=action_size,
                                         use_weightnorm=self._use_weightnorm,
                                         last_layer_init_func=lambda m: scale_layer_weights(m, init_noise, bias_offset=log_std_init),
                                         hidden_activations=inner_activations).to(device=torch_device)      

    def forward(self, observation_batch, reference_action : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        hidden_batch = self.act_fc(observation_batch)
        # dbg_check_finite(hidden_batch)
        mean = self.act_fc_mean(hidden_batch)
        if reference_action is not None:
            mean = mean + reference_action
        # The mean squashing does not alter the action probability, so no change should be necessary on
        # the logprob correction done in sample_action, I think
        # Still, it helps to avoid boudary issues with the noise being reduced on the edges of the action space
        mean_scales = self._mean_bounds_ratio*self.action_scale
        eps = 1e-6
        mean_scales = th.atanh(th.clamp(mean_scales, min=-1+eps, max=1-eps)) # because it gets squashed again later
        mean_biases = self.action_bias
        mean = th.tanh(mean/mean_scales)*mean_scales + mean_biases
        # ggLog.info(f"mean = {mean.min()} to {mean.max()}")

        log_std = self.act_fc_logstd(hidden_batch)
        log_std = (th.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min # squash the log_std network output
        log_std = log_std + th.log(self.action_scale) # scale the log_std with the action scale, so that it is relative to the action range
        return mean, log_std

    @th_compile_ext(mode=compile_mode, fullgraph=fullgraph, copy_outs=True, disable=disable_compile,  dynamic=dynamic_compile)
    def sample_action(self, observation_batch, reference_action : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        dbg_check_finite(observation_batch, async_assert=True, assert_msg="sac.Actor.sample_action: input is not finite")
        # observation_batch = map_tensor_tree(observation_batch, lambda t: t.fill_(0.0))
        batch_size = observation_batch.shape[0]
        mean, log_std = self(observation_batch, reference_action)
        std = log_std.exp()
        x_t = mean + th.empty_like(mean).normal_(mean=0.0, std=1.0)*std # rsample has issues with torch.compile
        normal = th.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t) # get the probability of the actions that we sampled
        y_t = th.tanh((x_t-self.action_bias)/self.action_scale) # squash the action in [-1,1]
        mean = th.tanh((mean -self.action_bias)/self.action_scale) # squash the mean in [-1,1] in the same way as the action

        # scale mean and action to the proper bounds
        # mean = th.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = log_prob - th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # correct the probability for the squashing and scaling
        log_prob = log_prob.sum(dim=1) # get probability per each multidimensional action, not for each action component

        dbg_check_finite(action, async_assert=True, assert_msg="sac.Actor.sample_action: action is not finite")
        dbg_check_size(action,   (batch_size, self._action_size), f"sac.Actor.sample_action: action has incorrect size {action.size()}")
        dbg_check_size(log_prob, (batch_size, ), f"sac.Actor.sample_action: log_prob has incorrect size {log_prob.size()}")
        return action, log_prob, mean, log_std

class SAC(RLAgent):
    @dataclass
    class Hyperparams():
        action_init : float | th.Tensor
        action_max : th.Tensor
        action_min : th.Tensor
        action_reference_obs_key : str | None
        action_size : int
        actor_observation_filter : list[str] | None
        actor_observation_space : gym.spaces.Space
        auto_entropy_temperature : bool
        batch_size : int
        constant_entropy_temperature : float | None
        critic_observation_filter : list[str] | None
        critic_observation_space : gym.spaces.Space
        feature_extractor_lr : float
        gamma : th.Tensor
        gamma_reward_scaling : bool
        log_std_init : float
        max_grad_norm : float
        log_alpha_grad_clip : float
        observation_space : gym.spaces.Space
        policy_arch : List[int]
        policy_lr : float
        policy_update_freq : int
        q_lr : float
        q_network_arch : List[int]
        target_entropy_annealing : tuple[Literal["constant", "ramp"], list[float | th.Tensor]] | AnnealingFunction |None
        target_entropy_factor : float
        target_tau : float
        targets_update_freq : int
        torch_device : th.device
        critic_weight_decay : float
        actor_weight_decay : float
        actor_mean_bounds_ratio : float
        rewards_num : int
        alpha_lr_factor : float
        reward_space : spaces.ThBox
        independent_entropy_q : bool
        perform_multiple_actor_updates : bool
        init_hparams : SAC_init_hparams

    def __init__(self,
                 init_hparams : SAC_init_hparams,
                 observation_space : gym.spaces.Space,
                 reward_space : gym.spaces.Space,
                 action_space : gym.spaces.Box,
                 actor_feature_extractor : FeatureExtractor | None = None,
                 critic_feature_extractor : FeatureExtractor | None = None                 ):
        super().__init__()
        self._init_args = get_func_input_args(exclude=[ "self",
                                                        "values",
                                                        "__class__",
                                                        "critic_feature_extractor",
                                                        "actor_feature_extractor"])
        # ggLog.info(f"self._init_args = \n"+pprint.pformat(self._init_args))
        action_size=int(np.prod(action_space.shape))
        action_min = action_space.low.tolist()
        action_max = action_space.high.tolist()
        action_init=action_space.zero_action if isinstance(action_space,spaces.ThBox) else 0.0
        self._init_args = copy.deepcopy(self._init_args)
        if not isinstance(reward_space, spaces.gym_spaces.Box):
            raise RuntimeError(f"SAC currently only supports ThBox reward spaces, but got {type(reward_space)}")
        self._env_reward_space = reward_space
        rnd_hparams = init_hparams.rnd_hyperparams
        rnd_reward_channel_added = (rnd_hparams is not None and
                                    rnd_hparams.scaler_hyperparams.separate_reward_channel)
        if rnd_reward_channel_added:
            # Widen the reward vector before rewards_num, the gammas and the q names are derived
            # from it, as those fix the critic's output size for the life of the agent.
            reward_space = _append_reward_channel(reward_space,
                                                  name = rnd_hparams.reward_channel_name,
                                                  torch_device = init_hparams.model_th_device)
        self._reward_space = reward_space
        rewards_num = spaces.get_1d_space_size(reward_space)
        reward_names = reward_space.labels if isinstance(reward_space, spaces.ThBox) else np.array([f"reward_r{i:03d}" for i in range(rewards_num)], dtype=object)
        if reward_names.ndim == 0:
            reward_names = np.expand_dims(reward_names, axis=0)
        init_hparams = copy.deepcopy(init_hparams)
        if init_hparams.target_entropy_factor is None:
            init_hparams.target_entropy_factor = -1.0
        actor_observation_space = filter_dict_space(observation_space, init_hparams.actor_observation_filter)
        critic_observation_space = filter_dict_space(observation_space, init_hparams.critic_observation_filter)
        if init_hparams.action_reference_obs_key is not None:
            action_init = 0.0

        gammas = init_hparams.gamma
        ggLog.info(f"reward_names = {reward_names}")
        
        if isinstance(gammas, Mapping):
            if rnd_reward_channel_added and rnd_hparams.reward_channel_name not in gammas:
                # The novelty channel is appended by SAC, so the caller's mapping does not know it.
                gammas = dict(gammas)
                gammas[rnd_hparams.reward_channel_name] = (rnd_hparams.gamma
                                                           if rnd_hparams.gamma is not None
                                                           else max(float(g) for g in gammas.values()))
            gammas = th.as_tensor([gammas[rn] for rn in reward_names], dtype=th.float32)
        elif isinstance(gammas, th.Tensor):
            gammas = gammas.expand(rewards_num).to(device=init_hparams.model_th_device)
        elif isinstance(gammas, float):
            gammas = th.as_tensor(gammas).expand(rewards_num).to(device=init_hparams.model_th_device)
        else:
            raise RuntimeError(f"Invalid gamma type {type(init_hparams.gamma)}, must be float or th.Tensor or dict")
        if rnd_reward_channel_added and rnd_hparams.gamma is not None:
            # Exploration usually wants a shorter horizon than the task.
            gammas = th.cat([gammas[:-1].clone(),
                             th.as_tensor([rnd_hparams.gamma], dtype=gammas.dtype, device=gammas.device)])

        self._q_names = reward_names.tolist()
        if init_hparams.independent_entropy_q:
            entropy_gamma = th.mean(gammas) # Is this a reasonable choice? maybe expose it as a hyperparameter?
            gammas = th.cat([gammas, entropy_gamma.unsqueeze(0)], dim=0)
            self._q_names.append("entropy")
        dbg_check_size(gammas, (len(self._q_names),), msg=f"SAC: gammas size mismatch with rewards_num, is {gammas.size()} should be {len(self._q_names)}")

        self._dtype = th.float32
        self._hp = SAC.Hyperparams(q_lr=init_hparams.q_lr,
                                   policy_lr = init_hparams.policy_lr,
                                   gamma=gammas.to(device=init_hparams.model_th_device),
                                   auto_entropy_temperature=init_hparams.auto_entropy_temperature,
                                   constant_entropy_temperature=init_hparams.constant_entropy_temperature,
                                   action_init=action_init,
                                   action_size=action_size,
                                   action_min = th.as_tensor(action_min),
                                   action_max = th.as_tensor(action_max),
                                   target_tau = init_hparams.target_tau,
                                   policy_update_freq=init_hparams.policy_update_freq,
                                   targets_update_freq=init_hparams.target_update_freq,
                                   q_network_arch = init_hparams.q_network_arch,
                                   policy_arch = init_hparams.policy_arch,
                                   torch_device = th.device(init_hparams.model_th_device),
                                   target_entropy_factor = init_hparams.target_entropy_factor,
                                   observation_space = observation_space,
                                   feature_extractor_lr = init_hparams.feature_extractor_lr,
                                   batch_size = init_hparams.batch_size,
                                   max_grad_norm=init_hparams.max_grad_norm,
                                   log_alpha_grad_clip = init_hparams.log_alpha_grad_clip,
                                   log_std_init = init_hparams.actor_log_std_init,
                                   actor_observation_space = actor_observation_space,
                                   critic_observation_space = critic_observation_space,
                                   actor_observation_filter = init_hparams.actor_observation_filter,
                                   critic_observation_filter = init_hparams.critic_observation_filter,
                                   target_entropy_annealing = init_hparams.target_entropy_factor_annealing,
                                   action_reference_obs_key = init_hparams.action_reference_obs_key,
                                   critic_weight_decay = init_hparams.critic_weight_decay,
                                   actor_weight_decay = init_hparams.actor_weight_decay,
                                   actor_mean_bounds_ratio = init_hparams.actor_mean_bounds_ratio,
                                   rewards_num = rewards_num,
                                   gamma_reward_scaling = True,
                                   alpha_lr_factor=init_hparams.alpha_lr_factor,
                                   reward_space = reward_space,
                                   independent_entropy_q = init_hparams.independent_entropy_q,
                                   init_hparams = init_hparams,
                                   perform_multiple_actor_updates = True)
        self._transition_augmentor_func = None
        self._reward_augmentor_func : RewardAugmentorProtocol | None = None
        self._postupdate_hooks : list[PostUpdateHookProtocol] = []
        self._default_reward_weights = th.ones(rewards_num, dtype=self._dtype, device=self._hp.torch_device)
        self._obs_space_sizes = sizetree_from_space(observation_space)
        self.device = self._hp.torch_device
        self._critic_updates = 0
        self._alpha_updates = 0
        self._actor_updates = 0
        self._agent_updates = 0
        self._needs_target_update = True

        self._share_actor_critic_feature_extractor = (actor_feature_extractor==critic_feature_extractor and
                                                      init_hparams.actor_observation_filter==init_hparams.critic_observation_filter)
        # ggLog.info(f"SAC: independent_entropy_q = {self._hp.independent_entropy_q}")
        if self._share_actor_critic_feature_extractor:
            if critic_feature_extractor is None or actor_feature_extractor is None: # second considition is just for typing
                self._critic_feature_extractor = StackVectorsFeatureExtractor(observation_space=critic_observation_space,
                                                                   hp=StackVectorsFeatureExtractorInitArgs(device=self._hp.torch_device))
                self._actor_feature_extractor = self._critic_feature_extractor
            else:
                self._critic_feature_extractor = critic_feature_extractor
                self._actor_feature_extractor = actor_feature_extractor
        else:
            if critic_feature_extractor is None:
                self._critic_feature_extractor = StackVectorsFeatureExtractor(observation_space=critic_observation_space,
                                                                   hp=StackVectorsFeatureExtractorInitArgs(device=self._hp.torch_device))
            else:
                self._critic_feature_extractor = critic_feature_extractor
            if actor_feature_extractor is None:
                self._actor_feature_extractor = StackVectorsFeatureExtractor(observation_space=actor_observation_space,
                                                                   hp=StackVectorsFeatureExtractorInitArgs(device=self._hp.torch_device))
            else:
                self._actor_feature_extractor = actor_feature_extractor
        critic_input_size = self._critic_feature_extractor.encoding_size()
        actor_input_size = self._actor_feature_extractor.encoding_size()
        ggLog.info(f"SAC: inner critic_input_size = {critic_input_size}, inner actor_input_size = {actor_input_size}")
        self._q_net = QNetwork( observation_size = critic_input_size,
                                action_size = self._hp.action_size,
                                q_network_arch = self._hp.q_network_arch,
                                torch_device = self._hp.torch_device,
                                nets_num=2,
                                reward_space=self._hp.reward_space,
                                independent_entropy_q=self._hp.independent_entropy_q)
        self._q_net_target = QNetwork(  observation_size=critic_input_size,
                                        action_size=self._hp.action_size,
                                        q_network_arch=self._hp.q_network_arch,
                                        torch_device=self._hp.torch_device,
                                        nets_num=2,
                                        reward_space=self._hp.reward_space,
                                        independent_entropy_q=self._hp.independent_entropy_q)
        self._q_net_target.load_state_dict(self._q_net.state_dict())
        self._q_optimizer = AdamW(split_params_for_weight_decay(self._q_net, self._hp.critic_weight_decay), lr=self._hp.q_lr)
        self._actor = Actor(policy_arch=self._hp.policy_arch,
                            observation_size=actor_input_size,
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.torch_device,
                            log_std_init=self._hp.log_std_init,
                            action_mean_init=self._hp.action_init,
                            mean_bounds_ratio=self._hp.actor_mean_bounds_ratio,
                            dtype=self._dtype)
        # initial_actor_entropy = self._hp.action_size/2 * (math.log(2*math.pi)+1) + 0.5*math.log(self._hp.log_std_init**(2*self._hp.action_size))
        # self._actor_optimizer = optim.Adam(split_params_for_weight_decay(self._actor,self._hp.actor_weight_decay),
        #                                    lr=self._hp.policy_lr)
        self._base_target_entropy_factor = th.as_tensor(self._hp.target_entropy_factor, device=self._hp.torch_device, dtype=self._dtype)
        self._target_entropy = self._base_target_entropy_factor*self._hp.action_size
        if init_hparams.target_entropy_factor_annealing is None:
            init_hparams.target_entropy_factor_annealing = ("constant", [self._base_target_entropy_factor])
        if isinstance(init_hparams.target_entropy_factor_annealing, AnnealingFunction):
            self._target_entropy_factor_annealing = init_hparams.target_entropy_factor_annealing
        else:
            self._target_entropy_factor_annealing = annealings[init_hparams.target_entropy_factor_annealing[0]](*init_hparams.target_entropy_factor_annealing[1])
        if self._hp.auto_entropy_temperature:
            alpha_init = init_hparams.alpha_initial_value
            if alpha_init <= 0.0:
                alpha_init = 1e-8
            self._log_alpha = th.full((1,), math.log(alpha_init), requires_grad=True, device=init_hparams.model_th_device)
            self._alpha = self._log_alpha.exp().detach()
            # self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = th.as_tensor(self._hp.constant_entropy_temperature).to(device=self._hp.torch_device, non_blocking=self._hp.torch_device.type=="cuda")
            self._log_alpha = self._alpha.log().detach()
        alpha_lr = self._hp.q_lr * self._hp.alpha_lr_factor
        self._actor_and_alpha_optimizer = AdamW([{ "params":[self._log_alpha], "lr":alpha_lr}]+
                                                      split_params_for_weight_decay(self._actor,self._hp.actor_weight_decay,
                                                                                    extra_kwargs={"lr":self._hp.policy_lr}))

        if self._hp.feature_extractor_lr > 0:
            critic_extractor_params = list(self._critic_feature_extractor.parameters())
            if len(critic_extractor_params) > 0:
                self._critic_feature_extractor_optimizer = AdamW(critic_extractor_params, lr=self._hp.feature_extractor_lr)
            else:
                self._critic_feature_extractor_optimizer = None

            if self._share_actor_critic_feature_extractor:
                self._actor_feature_extractor_optimizer = self._critic_feature_extractor_optimizer
            else:
                actor_extractor_params = list(self._actor_feature_extractor.parameters())
                if len(actor_extractor_params) > 0:
                    self._actor_feature_extractor_optimizer = AdamW(actor_extractor_params, lr=self._hp.feature_extractor_lr)
                else:
                    self._actor_feature_extractor_optimizer = None
        else:
            self._critic_feature_extractor_optimizer = None
            self._actor_feature_extractor_optimizer = None
        
        # using an if seems to be the most efficient way to skip this
        self._enable_feature_extractor_training = self._critic_feature_extractor_optimizer is not None or self._actor_feature_extractor_optimizer is not None

        self._last_q_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_actor_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_alpha_loss = th.as_tensor(float("nan"), device=self.device)
        self._tot_grad_steps_count = 0
        self._enable_nvtx = True
        self._log_losses = False
        # This was optimized by improving GPU usage via cudagraphs, profiling with Nsight Systems
        # The profiling command was:
        #  sudo nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop \
        #    --cudabacktrace=true -x true --force-overwrite true -o my_profile -s cpu \
        #    virtualenv/lrjax/bin/python3 src/rreal/src/rreal/examples/half_cheetah.py --comment t --algo sac
        # Then the produced file can be drag and dropped into Nsight Systems GUI to see the profiling results (e.g. the timeline)        
        # Can still be optimized more, but some segments are tricky to include in th.compile and behave weird

        self._stats = { "tot_grad_steps_count":0,
                        "q_loss_tot":0.0,
                        "q_loss":0.0,
                        "actor_loss":0.0,
                        "alpha_loss":0.0,
                        "val_q_loss":0.0,
                        "val_actor_loss":0.0,
                        "val_alpha_loss":0.0,
                        "alpha":0.0}
        self._actor_stats_names = [ "avg_action_mean",   "min_action_mean",   "max_action_mean",   "q95_action_mean",   "q05_action_mean",
                                    "avg_action_logstd", "min_action_logstd", "max_action_logstd", "q95_action_logstd", "q05_action_logstd"]
        self._alpha_stats_names = ["avg_log_prob", "min_log_prob", "max_log_prob", "q95_log_prob", "q05_log_prob", "current_entropy"]
        self._stats.update({n:0.0 for n in self._actor_stats_names})
        example_q_stats = th.zeros((5, len(self._q_names)), device=self.device)
        self._update_q_stats(example_q_stats)

        log_folder = adarl.utils.session.default_session.log_folder()+"/sac_logs"
        os.makedirs(log_folder, exist_ok=True)
        if self._log_losses:
            self._losses_file = open(log_folder+"/losses.bin", "ab")


    def get_critic_encoding_size(self):
        return self._critic_feature_extractor.encoding_size()


    def get_actor_encoding_size(self):
        return self._actor_feature_extractor.encoding_size()


    def _nvtx_startup(self):
        if self._enable_nvtx and self._agent_updates == 20:
            th.cuda.cudart().cudaProfilerStart() #type: ignore


    def _nvtx_stop(self):
        if self._enable_nvtx and self._agent_updates > 30:
            th.cuda.cudart().cudaProfilerStop() #type: ignore


    def _nvtx_mark(self, name : str):
        if self._enable_nvtx:
            th.cuda.nvtx.mark(name)


    def _nvtx_start_range(self, name : str):
        if not hasattr(self, "_nvtx_range_stack"):
            self._stack_nvtx_range = []
        self._stack_nvtx_range.append(name)
        if self._enable_nvtx:
            th.cuda.nvtx.range_push(name)


    def _nvtx_end_range(self, name : str):
        if not hasattr(self, "_stack_nvtx_range") or len(self._stack_nvtx_range) == 0:
            raise RuntimeError("nvtx range stack is empty")
        last_name = self._stack_nvtx_range.pop()
        if last_name != name:
            raise RuntimeError(f"nvtx range stack mismatch, closing {name} currently in {last_name}")
        if self._enable_nvtx:
            th.cuda.nvtx.range_pop()


    def get_actor_subobservation(self, observation : DictObs)  -> DictObs:
        return filter_dict(observation, self._hp.actor_observation_filter)


    def get_critic_subobservation(self, observation : DictObs) -> DictObs:
        return filter_dict(observation, self._hp.critic_observation_filter)


    def _get_reference_action(self, observation_batch : DictObs) -> th.Tensor | None:
        if self._hp.action_reference_obs_key is not None:
            return observation_batch[self._hp.action_reference_obs_key]
        else:
            return None


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

            reference_data_folder = adarl.utils.session.default_session.run_info.get("reference_data_folder", None)
            if reference_data_folder is not None and os.path.isdir(reference_data_folder):
                for root, _, files in os.walk(reference_data_folder):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        arcname = os.path.join("reference_data",
                                               os.path.relpath(file_path, reference_data_folder))
                        archive.write(file_path, arcname=arcname)

    def load_(self, path : str):
        # Before loading the state dict we try to check that the models are compatible
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != self.__class__.__name__:
            raise RuntimeError(f"File was not saved by this class")
        equal, reasons = compare_dicts(self._init_args, extra["init_args"])
        if not equal:
            ggLog.warn("init args of loaded model differ from those of self.")
            load_yaml_args = yaml.dump(extra['init_args'])
            original_yaml_args = yaml.dump(self._init_args)
            ggLog.warn(f"self._init_args = \n{original_yaml_args}")
            ggLog.warn(f"load init_args  = \n{load_yaml_args}")
            ggLog.warn(f"Differing fields: \n{reasons}")

        self._critic_feature_extractor.check_init_args_match(
                                    extra["critic_feature_extractor_class_name"],
                                    extra["critic_feature_extractor_init_args"])
        self._actor_feature_extractor.check_init_args_match(
                                    extra["actor_feature_extractor_class_name"],
                                    extra["actor_feature_extractor_init_args"])
        with zipfile.ZipFile(path) as archive:
            with archive.open("sac.pth", "r") as sac_file:
                self.load_state_dict(th.load(sac_file))
            self._diff_reference_data(archive)

    def _diff_reference_data(self, archive : zipfile.ZipFile):
        # Compare the reference_data stored in the archive against the folder
        # currently indicated by the session's run_info, and warn if they differ.
        archived_files = {os.path.relpath(name, "reference_data") : name
                          for name in archive.namelist()
                          if name.startswith("reference_data/") and not name.endswith("/")}
        reference_data_folder = adarl.utils.session.default_session.run_info.get("reference_data_folder", None)
        if reference_data_folder is None or not os.path.isdir(reference_data_folder):
            if archived_files:
                ggLog.warn(f"Loaded model contains reference_data but run_info has no valid "
                           f"reference_data_folder (got {reference_data_folder!r}).")
            return

        current_files = {}
        for root, _, files in os.walk(reference_data_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                current_files[os.path.relpath(file_path, reference_data_folder)] = file_path

        only_in_archive = sorted(set(archived_files) - set(current_files))
        only_in_current = sorted(set(current_files) - set(archived_files))
        differing_content = []
        for relpath in sorted(set(archived_files) & set(current_files)):
            with archive.open(archived_files[relpath], "r") as f:
                archived_bytes = f.read()
            with open(current_files[relpath], "rb") as f:
                current_bytes = f.read()
            if archived_bytes != current_bytes:
                differing_content.append(relpath)

        if only_in_archive or only_in_current or differing_content:
            ggLog.warn(f"reference_data in loaded model differs from current "
                       f"reference_data_folder ({reference_data_folder}):\n"
                       f"  only in saved model:   {only_in_archive}\n"
                       f"  only in current folder: {only_in_current}\n"
                       f"  differing content:      {differing_content}")

    @override
    @classmethod
    def load(cls,   path : str,
                    device : th.device | None = None,
                    init_args_override : dict | None = None):
        with zipfile.ZipFile(path) as archive:
            with archive.open("extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class")
        init_args = extra["init_args"]
        if device is not None:
            init_args["init_hparams"].model_th_device = device
        if init_args_override is not None:
            init_args.update(init_args_override)
        with zipfile.ZipFile(path) as archive:
            critic_feature_extractor_class = get_feature_extractor(extra["critic_feature_extractor_class_name"])
            init_args["critic_feature_extractor"] = critic_feature_extractor_class.load(archive, name="critic_feature_extractor")
            if extra["share_feature_extractor"]:
                init_args["actor_feature_extractor"] = init_args["critic_feature_extractor"]
            else:
                actor_feature_extractor_class = get_feature_extractor(extra["actor_feature_extractor_class_name"])
                init_args["actor_feature_extractor"] = actor_feature_extractor_class.load(archive, name="actor_feature_extractor")
        override_struct(init_args, init_args_override)
        ggLog.info(f"SAC.load(): building model with args: \n"+pprint.pformat(init_args))
        model = SAC(**init_args)
        # At this point we should have a model that is initialized exactly like the one that was saved
        # So we can load into it the state from the checkpoint
        model.load_(path)
        return model

    @override
    def predict_action(self, observation_batch : DictObs , deterministic = False, extra_returns : dict | None = None):
        th.compiler.cudagraph_mark_step_begin()
        # s = {k:v.size() for k,v in observation.items()}
        # ggLog.info(f"predict: observation = {s}")


        # check if it is not a batch, if so, unsqueeze
        has_right_dims = map2_tensor_tree(observation_batch, self._obs_space_sizes,
                                            lambda obs,obs_space_size: obs.dim() == len(obs_space_size))
        is_batched = not all(flatten_tensor_tree(has_right_dims).values())
        if not is_batched:
            observation_batch = {k:t.unsqueeze(0) for k,t in observation_batch.items()}

        observation_batch = self.get_actor_subobservation(observation_batch)
        observation_batch = {k:t.to(device = self.device, non_blocking=self.device.type=="cuda") for k,t in observation_batch.items()}
        dbg_check_finite(observation_batch, async_assert=True, assert_msg="sac.predict_action: observation is not finite")
        observation_batch_enc = self._actor_feature_extractor.extract_features(observation_batch)
        reference_action = self._get_reference_action(observation_batch)
        action, log_prob, mean, log_std = self._actor.sample_action(observation_batch_enc, reference_action=reference_action)
        if not is_batched:
            action = action.squeeze()
            mean = mean.squeeze()
            log_prob = log_prob.squeeze()
        if extra_returns is not None:
            extra_returns["mean"] = mean
            extra_returns["log_prob"] = log_prob
            extra_returns["action"] = action
            extra_returns["log_std"] = log_std
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
    
    def _compute_batch_stats(self, batch : th.Tensor):
        mean = batch.mean(dim=0)
        min = batch.amin(dim=0)
        max = batch.amax(dim=0)
        q05 = safe_quantile(batch, 0.05)
        q95 = safe_quantile(batch, 0.95)
        return th.stack([mean, min, max, q05, q95], dim=0)

    @th.compile(mode=compile_mode, fullgraph=fullgraph, disable=disable_compile,  dynamic=dynamic_compile)
    def _compute_critic_loss(self,  transitions : DictTransitionBatch,
                                    get_stats : bool = True,
                                    critic_enc_obss : th.Tensor | None = None,
                                    critic_enc_next_obss : th.Tensor | None = None,
                                    actor_enc_next_obss : th.Tensor | None = None) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor], tuple[None, th.Tensor, th.Tensor, th.Tensor]]:
        critic_obss = self.get_critic_subobservation(transitions.observations)
        # actor_obss = self.get_actor_subobservation(transitions.observations)
        if critic_enc_obss is None:
            critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss)
        with th.no_grad():
            terminateds = transitions.terminated
            next_observations = transitions.next_observations
            actions = transitions.actions
            batch_size = terminateds.size()[0]
            dbg_check_size(terminateds, (batch_size, 1), "sac._compute_critic_loss: terminateds has incorrect size")
            q_size = self._hp.rewards_num+1 if self._hp.independent_entropy_q else self._hp.rewards_num

            actor_next_obss = self.get_actor_subobservation(next_observations)
            if critic_enc_next_obss is None:
                critic_next_obss = self.get_critic_subobservation(next_observations)
                critic_enc_next_obss = self._critic_feature_extractor.extract_features(critic_next_obss)   
            if self._share_actor_critic_feature_extractor:
                actor_enc_next_obss = critic_enc_next_obss
            else:
                if actor_enc_next_obss is None:
                    actor_enc_next_obss = self._actor_feature_extractor.extract_features(actor_next_obss)
            reference_action = self._get_reference_action(actor_next_obss)

            rewards = self._augment_rewards(transitions, critic_enc_obss, critic_enc_next_obss, actor_enc_next_obss)
            dbg_check_size(rewards, (batch_size, self._hp.rewards_num), "sac._compute_critic_loss: rewards has incorrect size")
            
            # Compute next-values for TD
            next_state_actions, next_state_log_pi, _, _ = self._actor.sample_action(actor_enc_next_obss, reference_action = reference_action)
            q_next = self._q_net_target.get_min_qval(critic_enc_next_obss, next_state_actions)
            
            dbg_check_size(q_next, (batch_size, q_size), "sac._compute_critic_loss: q_next has incorrect size")
            dbg_check_size(next_state_log_pi, (batch_size, ), "sac._compute_critic_loss: next_state_log_pi has incorrect size")
            
            reward_gammas = self._hp.gamma[:-1] if self._hp.independent_entropy_q else self._hp.gamma
            if self._hp.gamma_reward_scaling:
                # rewards = rewards * (1 - self._hp.gamma)
                rewards = rewards * (1 - reward_gammas)/(1-th.amax(reward_gammas)) # scale to keep similar reward magnitudes when using multiple

            # ggLog.info(f"sac._compute_critic_loss: independent_entropy_q = {self._hp.independent_entropy_q}")
            # ggLog.info(f"sac._compute_critic_loss: next_state_log_pi size = {next_state_log_pi.size()}")
            if not self._hp.independent_entropy_q:
                soft_q_next = q_next - self._alpha/self._hp.rewards_num * next_state_log_pi.view(batch_size,1)
                dbg_check_size(soft_q_next, (batch_size, self._hp.rewards_num), "sac._compute_critic_loss: soft_q_next has incorrect size")

                td_q_values = rewards + (1 - terminateds) * self._hp.gamma * soft_q_next
            else:
                # Keep the q normal, and put the entropy term separately in the last q element
                # so the only thing we must do is add the entropy term to the last q
                q_next[:, -1] += -self._alpha*next_state_log_pi
                r = th.zeros_like(q_next)
                r[:, :-1] = rewards
                r[:, -1] = 0.0
                td_q_values = r + (1 - terminateds) * self._hp.gamma * q_next



        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        q_values = self._q_net(critic_enc_obss, actions).view(batch_size,2,q_size)

        # ggLog.info(f"q_values.size() = {q_values.size()}")
        td_q_values = td_q_values.view(batch_size,1,q_size)
        td_q_values = td_q_values.expand(batch_size,2,q_size)
        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        square_errs = (q_values - td_q_values)**2
        per_reward_square_errs : th.Tensor = square_errs.mean(dim=(0,1)) # mean over batch and nets, batch x nets x rewards_num -> rewards_num


        with th.no_grad():
            q_stats = self._compute_batch_stats(th.amin(q_values.detach(), dim=1)) # get stats on min q values over the 2 networks
        stats = (per_reward_square_errs.detach().clone(), q_stats.detach())
        # if get_stats:
        #     with th.no_grad():
        #         q_stats = self._compute_batch_stats(th.amin(q_values.detach(), dim=1)) # get stats on min q values over the 2 networks
        #     stats = (per_reward_square_errs.detach().clone(), q_stats.detach())
        # else:
        #     stats = (None, None)
        encoded_obss = (None, critic_enc_obss, actor_enc_next_obss, critic_enc_next_obss)
        return th.sum(per_reward_square_errs), stats, encoded_obss

    @th.compile(mode=compile_mode, fullgraph=True, disable=disable_compile,  dynamic=dynamic_compile)
    def _critic_opt_step(self, q_loss : th.Tensor):
        simplified_clip_grad_norm_(list(self._q_net.parameters()), self._hp.max_grad_norm)
        self._q_optimizer.step()
        if self._needs_target_update:
            # self._nvtx_start_range("_update_target_nets")
            self._update_target_nets()
            # self._nvtx_end_range("_update_target_nets")
        self._last_q_loss.copy_(q_loss.detach())

    def _update_q_stats(self, q_val_stats : th.Tensor):
        if q_val_stats is not None:
            q_stat_by_rew = {self._q_names[i]: q_val_stats[:,i] for i in range(len(self._q_names))}
            for rew_name, stats in q_stat_by_rew.items():
                self._stats.update({f"q_val_{stat_name}_{rew_name}":stat_value for stat_name, stat_value in zip( ["avg", "min", "max", "q05", "q95"], stats.detach().clone())})
        
    def _update_critic(self, transitions : DictTransitionBatch):
        # ggLog.info(f"critic update...")
        
        # self._nvtx_start_range("_update_critic")
        self._q_optimizer.zero_grad(set_to_none=True)
        # self._nvtx_start_range("critic forward")
        # ggLog.info(f"compute_critic_loss...")
        q_loss, (subq_errs, q_stats), enc_obss = self._compute_critic_loss(transitions)
        # ggLog.info(f"compute_critic_loss done")
        # self._nvtx_end_range("critic forward")
        # self._nvtx_start_range("critic backward")
        q_loss.backward()
        # self._nvtx_end_range("critic backward")
        # self._nvtx_start_range("critic opt")
        with th.no_grad():
            self._critic_opt_step(q_loss)
        
        if subq_errs is not None:            
            self._stats.update({f"q_loss_r_{self._q_names[i]}":err for i,err in enumerate(subq_errs.detach().clone())})
        self._update_q_stats(q_stats)
        # self._nvtx_end_range("critic opt")
        self._critic_updates += 1
        # self._nvtx_end_range("_update_critic")
        # ggLog.info(f"critic update done")
        return enc_obss


    # @th.compile(mode=compile_mode, fullgraph=fullgraph)
    def _compute_actor_loss(self,   transitions : DictTransitionBatch,
                                    freeze_critic : bool = False,
                                    get_stats : bool = False,
                                    actor_enc_obss : th.Tensor | None = None,
                                    critic_enc_obss : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor | None]:
        # self._nvtx_mark("_compute_actor_loss")
        batch_size = transitions.rewards.shape[0]
        actor_obss = self.get_actor_subobservation(transitions.observations)
        # self._nvtx_mark("actor_enc")
        if actor_enc_obss is None:
            actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss)
        # self._nvtx_mark("get ref")
        if self._share_actor_critic_feature_extractor:
            critic_enc_obss = actor_enc_obss
        else:
            if critic_enc_obss is None:
                critic_obss = self.get_critic_subobservation(transitions.observations)
                critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss)
        
        reference_action = self._get_reference_action(actor_obss)
        
        # self._nvtx_mark("sample")
        act, act_log_prob, act_mean, act_logstd = self._actor.sample_action(actor_enc_obss, reference_action=reference_action)
        # act, act_log_prob = act.clone(), act_log_prob.clone() # prevent issues with cuda graphs
        # with th.no_grad():
        # self._nvtx_mark("crit_enc")
        # self._nvtx_mark("get_q")
        if freeze_critic: 
            # Needed if we are updating actor and critic together, if done separately we just ignore these grads at optimizer time
            # In torch compile we cannot change requires grad, so we detach the weights, using this functional thing
            min_qs_pi = th.amin(th.func.functional_call(self._q_net, {k:t.detach() for k,t in dict(self._q_net.named_parameters()).items()}, (critic_enc_obss, act)), dim = 1) 
        else:    
            min_qs_pi = self._q_net.get_min_qval(critic_enc_obss, act)
        min_qs_pi = min_qs_pi.sum(dim=1) # Sum all the qvalues for the different rewards
        
        dbg_check_size(min_qs_pi, (batch_size,), f"sac._compute_actor_loss: min_qs_pi has incorrect size {min_qs_pi.size()}")
        dbg_check_size(act_log_prob, (batch_size,), f"sac._compute_actor_loss: act_log_prob has incorrect size {act_log_prob.size()}")

        # self._nvtx_mark("ret_q")
        if get_stats:
            actor_stats = th.stack([act_mean.mean(),
                                    act_mean.min(),
                                    act_mean.max(),
                                    safe_quantile(act_mean, 0.95),
                                    safe_quantile(act_mean, 0.05),
                                    act_logstd.mean(),
                                    act_logstd.min(),
                                    act_logstd.max(),
                                    safe_quantile(act_logstd, 0.95),
                                    safe_quantile(act_logstd, 0.05)])
        else:
            actor_stats = None
        return ((self._alpha * act_log_prob) - min_qs_pi).mean(), actor_stats
    
    # @th.compile(mode=compile_mode, fullgraph=fullgraph)
    def _alpha_loss(self, act_log_prob : th.Tensor):
        # current_entropy = -act_log_prob.mean()
        # # if current_entropy > target_entropy then alpha goes toward zero (focus on reward maximization)
        # # if current_entropy < target_entropy then alpha goes toward +inf (focus on entropy maximization) (maybe cap it?)
        # return self._log_alpha.exp() * (current_entropy - self._target_entropy) 
        return (-self._log_alpha.exp() * (act_log_prob + self._target_entropy)).mean()
    
    # @th.compile(mode=compile_mode, fullgraph=fullgraph)
    def _alpha_stats(self, act_log_prob : th.Tensor):
        return th.stack([   act_log_prob.mean(),
                            act_log_prob.min(),
                            act_log_prob.max(),
                            safe_quantile(act_log_prob, 0.95), # has issues with dynamic compiles (which torch may decide to do sometimes)
                            safe_quantile(act_log_prob, 0.05),
                            -act_log_prob.mean()] )

    @th.compile(mode=compile_mode, fullgraph=fullgraph, disable=disable_compile,  dynamic=dynamic_compile)
    def _compute_alpha_loss(self, transitions : DictTransitionBatch,
                            actor_enc_obss : th.Tensor | None = None):
        with th.no_grad():
            actor_obss = self.get_actor_subobservation(transitions.observations)
            if actor_enc_obss is None:
                actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss)
            reference_action = self._get_reference_action(actor_obss)
            _, act_log_prob, _, _ = self._actor.sample_action(actor_enc_obss, reference_action=reference_action)
            stats = self._alpha_stats(act_log_prob)
        return self._alpha_loss(act_log_prob), stats
    

    @th.compile(mode=compile_mode, fullgraph=fullgraph, disable=disable_compile, dynamic=dynamic_compile)
    def _compute_actor_and_alpha_loss(self, transitions : DictTransitionBatch):
        # precompute actor encodings to save time
        actor_enc_obss = self._actor_feature_extractor.extract_features(self.get_actor_subobservation(transitions.observations))
        actor_loss, actor_stats = self._compute_actor_loss(transitions, get_stats=False,
                                                           actor_enc_obss=actor_enc_obss)
        # self._start_range_nvtx("actor opt")
        
        # self._start_range_nvtx("_update_alpha")
        if self._hp.auto_entropy_temperature:
            alpha_loss, alpha_stats = self._compute_alpha_loss(transitions,
                                                               actor_enc_obss=actor_enc_obss)
            loss = actor_loss + alpha_loss
        else:
            alpha_loss, alpha_stats = None, None
            loss = actor_loss
        
        return loss, actor_loss, alpha_loss, alpha_stats, actor_stats
    
    def _compute_encodings(self, transitions : DictTransitionBatch):
        critic_enc_obss = self._critic_feature_extractor.extract_features(self.get_critic_subobservation(transitions.observations))
        critic_enc_obss = critic_enc_obss.clone() # clone to avoid issues with cudagraphs
        if self._share_actor_critic_feature_extractor:
            actor_enc_obss = critic_enc_obss
        else:
            actor_enc_obss = self._actor_feature_extractor.extract_features(self.get_actor_subobservation(transitions.observations))
            actor_enc_obss = actor_enc_obss.clone()
        with th.no_grad():
            critic_enc_next_obss = self._critic_feature_extractor.extract_features( self.get_critic_subobservation(transitions.next_observations))
            critic_enc_next_obss = critic_enc_next_obss.clone()
            if self._share_actor_critic_feature_extractor:
                actor_enc_next_obss = critic_enc_next_obss
            else:
                actor_enc_next_obss = self._actor_feature_extractor.extract_features(self.get_actor_subobservation(transitions.next_observations))
                actor_enc_next_obss = actor_enc_next_obss.clone()
        return actor_enc_obss, critic_enc_obss, actor_enc_next_obss, critic_enc_next_obss
    
    @th.compile(mode=compile_mode, fullgraph=fullgraph, disable=disable_compile,  dynamic=dynamic_compile)
    def _compute_all_losses(self, transitions):
        # Precompute encodings to save time
        actor_enc_obss, critic_enc_obss, actor_enc_next_obss, critic_enc_next_obss = self._compute_encodings(transitions)
        q_loss, (subq_square_errs, q_stats), encoded_obss = self._compute_critic_loss(transitions,
                                                                        critic_enc_obss=critic_enc_obss,
                                                                        critic_enc_next_obss=critic_enc_next_obss,
                                                                        actor_enc_next_obss=actor_enc_next_obss)
        actor_loss, actor_stats = self._compute_actor_loss(transitions, freeze_critic=True, get_stats=False,
                                                            actor_enc_obss=actor_enc_obss,
                                                            critic_enc_obss=critic_enc_obss)
        
        if self._hp.auto_entropy_temperature:
            alpha_loss, alpha_stats = self._compute_alpha_loss(transitions,
                                                               actor_enc_obss=actor_enc_obss)
            loss = actor_loss + alpha_loss + q_loss
        else:
            alpha_loss, alpha_stats = th.zeros_like(actor_loss), None
            loss = actor_loss + q_loss
        encoded_obss = (actor_enc_obss, critic_enc_obss, actor_enc_next_obss, critic_enc_next_obss)
        return loss, q_loss, actor_loss, alpha_loss, alpha_stats, actor_stats, (subq_square_errs, q_stats), encoded_obss
    
    @th.compile(mode=compile_mode, fullgraph=True, disable=disable_compile,  dynamic=dynamic_compile)
    def _actor_and_alpha_opt_step(self, actor_loss : th.Tensor, alpha_loss : th.Tensor | None):
        simplified_clip_grad_norm_(list(self._actor.parameters()), self._hp.max_grad_norm)
        clipped_log_alpha_grad : th.Tensor = th.clamp(self._log_alpha.grad, -self._hp.log_alpha_grad_clip, self._hp.log_alpha_grad_clip)
        self._log_alpha.grad.copy_(clipped_log_alpha_grad)
        self._actor_and_alpha_optimizer.step()
        self._alpha.fill_(self._log_alpha.exp().detach().view(tuple())) # keep the same address to make cudagraphs happy
        self._last_actor_loss.copy_(actor_loss.detach())
        if alpha_loss is not None:
            self._last_alpha_loss.copy_(alpha_loss.detach())


    @th.compile(mode=compile_mode, fullgraph=True, disable=disable_compile,  dynamic=dynamic_compile)
    def _all_opt_step(self, q_loss : th.Tensor,
                            actor_loss : th.Tensor,
                            alpha_loss : th.Tensor):
        self._critic_opt_step(q_loss)
        self._actor_and_alpha_opt_step(actor_loss, alpha_loss)

    def _update_actor_and_alpha(self, transitions : DictTransitionBatch):
        # We aggregate actor and alpha to join the two compilation regions and cuda graphs, so to reduce overhead
        # self._nvtx_start_range("_update_actor_and_alpha")
        self._actor_and_alpha_optimizer.zero_grad(set_to_none=True)
        # self._nvtx_start_range("_compute_actor_and_alpha_loss")
        actor_alpha_loss, actor_loss, alpha_loss, alpha_stats, actor_stats = self._compute_actor_and_alpha_loss(transitions)
        # self._nvtx_end_range("_compute_actor_and_alpha_loss")
        # self._nvtx_start_range("actor_alpha backward")
        actor_alpha_loss.backward()
        # self._nvtx_end_range("actor_alpha backward")
        # self._nvtx_start_range("actor_alpha_opt")
        with th.no_grad():
            self._actor_and_alpha_opt_step(actor_loss, alpha_loss)
        # self._nvtx_end_range("actor_alpha_opt")
        if alpha_stats is not None:
            self._stats.update({k:v for k,v in zip(self._alpha_stats_names,alpha_stats.detach().clone())})
        if actor_stats is not None:
            self._stats.update({k:v for k,v in zip(self._actor_stats_names,actor_stats.detach().clone())})      
        self._alpha_updates += 1
        self._actor_updates += 1
        # self._nvtx_end_range("_update_actor_and_alpha")

    def _update_all(self, transitions):
        # self._nvtx_start_range("_update_all")
        # raise NotImplementedError("This does not work for some reason")
        self._actor_and_alpha_optimizer.zero_grad(set_to_none=True)
        self._q_optimizer.zero_grad(set_to_none=True)

        # self._nvtx_start_range("_compute_all_losses")
        loss, q_loss, actor_loss, alpha_loss, alpha_stats, actor_stats, (subq_errs, q_stats), encoded_obss = self._compute_all_losses(transitions)
        # self._nvtx_end_range("_compute_all_losses")
        # self._nvtx_start_range("all_backward")
        loss.backward()
        # self._nvtx_end_range("all_backward")

        # self._nvtx_start_range("all_opt")
        with th.no_grad():
            self._all_opt_step(q_loss, actor_loss, alpha_loss)
        # self._nvtx_end_range("all_opt")

        if alpha_stats is not None:
            self._stats.update({k:v for k,v in zip(self._alpha_stats_names,alpha_stats.detach().clone())})
        if actor_stats is not None:
            self._stats.update({k:v for k,v in zip(self._actor_stats_names,actor_stats.detach().clone())})
        if subq_errs is not None:
            self._stats.update({f"q_loss_r_{self._q_names[i]}":err for i,err in enumerate(subq_errs.detach().clone())})
        self._update_q_stats(q_stats)
        self._critic_updates += 1
        self._alpha_updates += 1
        self._actor_updates += 1
        return encoded_obss
        # self._nvtx_end_range("_update_all")
        
    @staticmethod
    def _target_update(param, target_param, tau):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @th.compile(mode=compile_mode, fullgraph=fullgraph, disable=disable_compile,  dynamic=dynamic_compile)        
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

    def _update_full_merged(self, transitions : DictTransitionBatch):
        th.compiler.cudagraph_mark_step_begin()
        # self._nvtx_startup()
        nvtx_range_name = f"iteration{self._critic_updates}"
        # self._nvtx_start_range(nvtx_range_name)
        if self._enable_feature_extractor_training:
            if self._critic_feature_extractor_optimizer is not None:
                self._critic_feature_extractor_optimizer.zero_grad(set_to_none=True)
            if self._actor_feature_extractor_optimizer is not None:
                self._actor_feature_extractor_optimizer.zero_grad(set_to_none=True)
        update_actor_and_alpha = self._agent_updates % self._hp.policy_update_freq == 0
        if update_actor_and_alpha:
            encoded_obss = self._update_all(transitions)
            if self._hp.perform_multiple_actor_updates:
                for _ in range(self._hp.policy_update_freq-1): # do the remaining updates (twice on the same transition?? This comes from CleanRL)
                    self._update_actor_and_alpha(transitions=transitions)
        else:
            encoded_obss = self._update_critic(transitions)

        if self._enable_feature_extractor_training:
            self._update_feature_extractor()
        self._agent_updates += 1
        self._needs_target_update = self._critic_updates % self._hp.targets_update_freq == 0
        # self._nvtx_end_range(nvtx_range_name)
        # self._nvtx_stop()
        q_act_alpha_losses = (self._last_q_loss, self._last_actor_loss, self._last_alpha_loss)
        return q_act_alpha_losses, encoded_obss
    
    def validate(self, buffer : BaseValidatingBuffer, batch_size : int):
        with th.no_grad():
            transitions = buffer.sample_validation(batch_size=batch_size)
            critic_loss, (square_errs, q_stats), enc_obss = self._compute_critic_loss(transitions)
            actor_loss, _ = self._compute_actor_loss(transitions)
            alpha_loss, _ = self._compute_alpha_loss(transitions)
        self._stats.update({"val_q_loss":critic_loss,
                            "val_actor_loss":actor_loss,
                            "val_alpha_loss":alpha_loss})
        return critic_loss, actor_loss, alpha_loss


    def set_reward_augmentor_func(self, reward_augmentor_func : RewardAugmentorProtocol | None):
        """Sets a function to augment rewards using transitions and encoded observations, see RewardAugmentorProtocol. The function must be torch-compilable

        Parameters
        ----------
        reward_augmentor_func : RewardAugmentorProtocol | None
            The function to augment rewards, or None to disable reward augmentation.
        """
        self._reward_augmentor_func = reward_augmentor_func

        
    def _augment_rewards(self, transitions, critic_enc_obss, critic_enc_next_obss, actor_next_enc_obss):
        if self._reward_augmentor_func is None:
            return transitions.rewards
        else:
            return self._reward_augmentor_func(transitions,
                                               critic_enc_obss,
                                               critic_enc_next_obss,
                                               actor_next_enc_obss)
    

    def set_transition_augmentor(self, transition_augmentor: TransitionAugmentorFunction):
        self._transition_augmentor_func = transition_augmentor

    def _augment_transitions(self, transitions : DictTransitionBatch) -> DictTransitionBatch:
        if self._transition_augmentor_func is not None:
            o, a, no, r, t = self._transition_augmentor_func(transitions.observations,
                                                                transitions.actions,
                                                                transitions.next_observations,
                                                                transitions.rewards,
                                                                transitions.terminated)
            transitions = DictTransitionBatch(
                observations=o,
                actions=a,
                next_observations=no,
                rewards=r,
                terminated=t
            )
        return transitions

    def register_postupdate_hook(self, hook_func : PostUpdateHookProtocol):
        self._postupdate_hooks.append(hook_func)

    def _run_postupdate_hooks(self, transitions : DictTransitionBatch,
                              encoded_obss : tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
                              losses : tuple[th.Tensor, th.Tensor, th.Tensor]):
        detached_encoded_obss = (encoded_obss[0].detach() if encoded_obss[0] is not None else None,
                                encoded_obss[1].detach(),
                                encoded_obss[2].detach(),
                                encoded_obss[3].detach())
        detached_losses = (losses[0].detach(), losses[1].detach(), losses[2].detach())
        all_logs = {}
        for hook in self._postupdate_hooks:
            logs = hook(transitions, detached_encoded_obss, detached_losses)
            all_logs.update(logs)
        return all_logs

    @override
    def train_model(self, global_step, iterations, buffer : BaseBuffer) -> tuple[th.Tensor,th.Tensor,th.Tensor]:
        # ggLog.info(f":::::::::::::::::::::::: train_model: global_step={global_step}")
        qloss_actloss_alphaloss_alpha = [None]*iterations
        hooks_logs = {}
        target_entropy_cpu = self._target_entropy_factor_annealing(global_step, self._tot_grad_steps_count)*self._hp.action_size
        self._target_entropy.copy_(th.as_tensor(target_entropy_cpu).to(device=self.device, dtype=self._dtype, non_blocking=self.device.type=="cuda"))
        t0 = time.monotonic()
        for i in range(iterations):
            # self._nvtx_start_range("sample")
            transitions : DictTransitionBatch = buffer.sample(self._hp.batch_size) #TODO: maybe add a check that does this cast better

            transitions = self._augment_transitions(transitions)

            # self._nvtx_end_range("sample")
            # transitions = map_tensor_tree(transitions, lambda t : t.to(device=self.device, non_blocking=self.device.type=="cuda"))
            # th.cuda.synchronize(self.device)
            losses, encoded_obss = self._update_full_merged(transitions = transitions)

            hooks_logs.update(self._run_postupdate_hooks(transitions, encoded_obss, losses))

            qloss_actloss_alphaloss_alpha[i] = losses + (self._alpha,)
            self._tot_grad_steps_count += 1
        t1 = time.monotonic()
        # q_loss, actor_loss, alpha_loss = th.as_tensor(q_act_alpha_losses).mean(dim = 0).cpu().numpy()
        if iterations > 0:
            if self._log_losses:
                np.array(th.as_tensor(qloss_actloss_alphaloss_alpha, dtype=th.float32).cpu().numpy(), dtype=np.float32).tofile(self._losses_file)
                self._losses_file.flush()
            q_loss, actor_loss, alpha_loss, alpha = qloss_actloss_alphaloss_alpha[-1]
        else:
            with th.no_grad():
                # compute losses but don't train
                transitions = buffer.sample(batch_size=self._hp.batch_size)
                q_loss, (square_errs, q_stats), enc_obss = self._compute_critic_loss(transitions)
                actor_loss, _ = self._compute_actor_loss(transitions)
                alpha_loss, _ = self._compute_alpha_loss(transitions)
        self._stats.update({"tot_grad_steps_count":self._tot_grad_steps_count,
                            "q_loss_tot":q_loss,
                            "q_loss":q_loss,
                            "actor_loss":actor_loss,
                            "alpha_loss":alpha_loss,
                            "alpha":self._alpha.clone(),
                            "target_entropy":target_entropy_cpu,
                            "iterations_per_second":iterations/(t1-t0)})
        # ggLog.info(f"hook logs = {hooks_logs}")
        self._stats.update(hooks_logs)
        return q_loss, actor_loss, alpha_loss

    @override
    def input_device(self):
        return self._hp.torch_device
    
    def get_stats(self):
        return self._stats
    
    @override
    def get_reference_init_args(self) -> dict:
        return self._hp.init_hparams.reference_init_args

register_agent_class(SAC)

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
                    validation_batch_size : int = 256,
                    parallelize_experience_collection : bool = True):
    if validation_freq>0 and not isinstance(buffer, BaseValidatingBuffer):
        raise RuntimeError(f"validation_freq>0 but buffer is not a BaseValidatingBuffer")
    if log_freq_vstep == -1: log_freq_vstep = train_freq
    num_envs = collector.num_envs()

    collector.reset()
    global_exp_step = 0
    t_train_sl, t_coll_sl, t_tot_sl, steps_sl, t_val_sl, t_buff_sl, t_add_sl, t_start_sl, t_end_callbacks_sl,t_wait_collect_sl = 0,0,0,0,0,0,0,0,0,0
    
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
    grad_steps_done_sl = 0
    q_loss, actor_loss, alpha_loss = th.as_tensor(float("nan")),th.as_tensor(float("nan")),th.as_tensor(float("nan"))
    start_time = time.monotonic()
    last_log_steps = float("-inf")

    session = adarl.utils.session.default_session
    # th.cuda.memory._record_memory_history(max_entries=100_000)
    while global_exp_step < total_timesteps and not session.is_shutting_down():
        s0b = buffer.collected_frames()
        t0 = time.monotonic()

        # ------------------  Start experience collection  ------------------
        steps_to_collect = train_freq*num_envs
        vsteps_to_collect = train_freq
        callbacks.on_collection_start()
        collector.start_collection(model_state_dict=model.state_dict(),
                                            vsteps_to_collect=vsteps_to_collect,
                                            global_vstep_count=global_exp_step//num_envs,
                                            random_vsteps=learning_start_step//num_envs)
        if not parallelize_experience_collection:
            tmp_buff = collector.wait_collection(timeout = 300.0)
        # ------------------             Train             ------------------
        t_before_train = time.monotonic()
        trained = False
        grad_steps_done = 0
        if global_exp_step > learning_start_step:
            iterations = grad_steps if grad_steps!="auto" else 10
            while (grad_steps != "auto" and not trained) or (grad_steps == "auto" and collector.is_collecting()):
                trained = True
                q_loss, actor_loss, alpha_loss = model.train_model(global_exp_step, iterations, buffer)
                grad_steps_done += iterations
                session.run_info["train_iterations"].value = session.run_info["train_iterations"].value + iterations

            train_count += 1
        t_after_train = time.monotonic()
        if trained and validation_freq>0 and train_count%validation_freq==0:
            if not isinstance(buffer, BaseValidatingBuffer):
                raise RuntimeError(f"validation_freq>0 but buffer is not a BaseValidatingBuffer")
            model.validate(buffer, batch_size=validation_batch_size)
        t_after_val = time.monotonic()
        
        # ------------------   Store collected experience  ------------------
        if parallelize_experience_collection:
            tmp_buff = collector.wait_collection(timeout = 300.0)
        t_after_wait = time.monotonic()
        new_episodes = tmp_buff.added_completed_episodes() - ep_counter
        ep_counter = tmp_buff.added_completed_episodes()
        step_counter = tmp_buff.added_frames()
        t_coll_sl += collector.collection_duration()
        session.run_info["collected_episodes"].value = ep_counter
        session.run_info["collected_steps"].value = step_counter
        # callbacks._callbacks[0].set_model(model)
        callbacks.on_collection_end(collected_steps=vsteps_to_collect*num_envs,
                                   collected_episodes=new_episodes,
                                   collected_data=tmp_buff)
        t_after_endcallback = time.monotonic()
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
        global_exp_step += steps_to_collect
        steps_sl += steps_to_collect
        tf = time.monotonic()
        t_start_sl              += t_before_train       - t0
        t_train_sl              += t_after_train        - t_before_train
        t_val_sl                += t_after_val          - t_after_train
        t_wait_collect_sl       += t_after_wait         - t_after_val
        t_end_callbacks_sl      += t_after_endcallback  - t_after_wait
        t_buff_sl               += t_after_buff         - t_after_endcallback
        t_add_sl                += t_add
        t_tot_sl                += tf-t0
        grad_steps_done_sl += grad_steps_done
        iter_per_sec = grad_steps_done_sl/t_train_sl
        t = time.monotonic()
        # ggLog.info(f"global_steps = {global_step}")
        if trained:
            # ggLog.info(f"SAC: "+str([f"{k}={v}, " for k,v in model.get_stats().items()]))
            wlogs = {"sac/"+k:v for k,v in model.get_stats().items()}
            # ggLog.info(f"Wandb log: "+str(wlogs.keys()))
            wlogs["sac/ips"] = iter_per_sec
            wlogs["sac/buffer_frames"] = buffer.stored_frames()
            wlogs["sac/val_buffer_frames"] = buffer.stored_validation_frames() if isinstance(buffer,BaseValidatingBuffer) else 0
            # ggLog.info(f"SAC Wandb log has q_val_avg={wlogs['sac/q_val_avg']}")
            wandb_log(wlogs,throttle_period=2, silent_throttling=True)
        if global_exp_step - last_log_steps > log_freq_vstep*num_envs:
            last_log_steps = global_exp_step
            ips = model.get_stats().get('iterations_per_second',float("nan"))
            log_async(f"SAC: expsteps={global_exp_step}"+" q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}"+f" ips={ips:.2f}",
                      tensors=dict(q_loss=q_loss,actor_loss=actor_loss,alpha_loss=alpha_loss))
            # ggLog.info(f"SAC: expsteps={global_step} q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}")
            ggLog.info(f"OFFTRAIN: expstps:{global_exp_step}"
                       f" trainstps={model._tot_grad_steps_count}"
                    #    f" exp_reuse={model._tot_grad_steps_count*batch_size/global_step:.2f}"
                       f" tcoll={t_coll_sl:.2f}"
                       f" train={t_train_sl:.2f}"
                       f" tstrt={t_start_sl:.2f}"
                       f" tval={t_val_sl:.2f}"
                       f" twait={t_wait_collect_sl:.2f}"
                       f" tce={t_end_callbacks_sl:.2f}"
                       f" tbuff={t_buff_sl:.2f}"
                       f" tadd={t_add_sl:.2f}"
                       f" tot={t_tot_sl:.2f}"
                       f" fps={steps_sl/t_tot_sl:.2f} collfps={steps_sl/t_coll_sl:.2f}"
                       f" ips={iter_per_sec:.2f}"
                       f" alltime_fps={global_exp_step/(t-start_time):.2f} alltime_ips={model._tot_grad_steps_count/(t-start_time):.2f}")
            dictlist = [f"{k}:{v:.6g}" for k,v in collector.get_stats().items()]
            ggLog.info(f"Collection: {', '.join(dictlist)}")
            t_train_sl, t_coll_sl, t_tot_sl, steps_sl, t_val_sl, t_buff_sl, t_add_sl, t_start_sl, t_end_callbacks_sl, t_wait_collect_sl, grad_steps_done_sl = 0,0,0,0,0,0,0,0,0,0,0
            free, total = th.cuda.mem_get_info(th.device('cuda:0'))
            mem_used_MB = (total - free) / 1024 ** 2
            # ggLog.info(f"{t}: cuda mem usage = {mem_used_MB}")
        # jax.profiler.save_device_memory_profile(f"jax_memory_{t}.prof")
        # th.cuda.memory._dump_snapshot(f"memory_{t}_th.pickle")
        adarl.utils.sigint_handler.haltOnSigintReceived()
    ggLog.info(f"Off-policy train terminating...")
    callbacks.on_training_end()
    ggLog.info(f"Off-policy train terminated.")


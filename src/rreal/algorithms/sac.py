
from __future__ import annotations
from gc import freeze

from adarl.utils.buffers import ThDReplayBuffer, TransitionBatch, BaseBuffer, BaseValidatingBuffer
from adarl.utils.callbacks import TrainingCallback, CallbackList
from adarl.utils.tensor_trees import sizetree_from_space, map2_tensor_tree, flatten_tensor_tree, map_tensor_tree
from adarl.utils.wandb_wrapper import wandb_log
from adarl.utils.dbg.dbg_checks import dbg_check_finite
from adarl.utils.utils import get_func_input_args, th_compile_ext
from dataclasses import dataclass, asdict
from rreal.algorithms.collectors import ExperienceCollector
from rreal.algorithms.rl_agent import RLAgent
from rreal.feature_extractors import get_feature_extractor
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor
from rreal.utils import build_mlp_net, scale_layer_weights, split_params_for_weight_decay, simplified_clip_grad_norm_
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
from typing import Protocol

th._dynamo.config.compiled_autograd = True

# There seems to be a BIG overhead in entering end exiting compiled functions
# I tried to dig a bit in the torch/dynamo/inductor code to understand at the end what is being ran when
# calling a compiled function, but its deeeeeeeep
compile_mode="max-autotune" # reduce overhead doesn't seem to reduce overhead more than max-autotune

def nop_func(arg1):
    pass
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
    device : str | th.device
    """The torch device where the model will be located"""
    gamma : float
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
    target_entropy_factor_annealing : tuple[Literal['constant', 'ramp'], list[float | th.Tensor]] | None = None
    """The target entropy factor annealing function, by default it is None (no annealing), see predefined annealings in sac.py"""
    action_reference_obs_key : str | None = None
    """The observation key that will be used as a reference for the action, meaning the actor distribution is computed as `mean = NN(obs) + act_ref` 
      by default it is None (no reference, the mean is produced from the network directly)"""
    max_grad_norm : float = 0.5
    feature_extractor_lr : float = 0.0
    torch_device : Union[str,th.device] = "cuda"
    policy_update_freq : int = 2
    target_update_freq : int = 1
    auto_entropy_temperature : bool =True
    constant_entropy_temperature : float | None =None
    critic_weight_decay : float = 0.0
    actor_weight_decay : float = 0.0
    deterministic_collection_ratio : float = 0.0


class AnnealingFunction(Protocol):
        def __call__(self,  global_step : int, iterations : int) -> float:
            ...
def get_constant_annealing(value : float) -> AnnealingFunction:
    """
    Returns a function that always returns the same value.
    This is used for the target entropy in SAC.
    """
    def constant_annealing(global_step : int, iterations : int) -> float:
        return value
    return constant_annealing

def get_ramp_annealing(ramp_start_step : int, ramp_end_step : int, start_value : float, end_value : float) -> AnnealingFunction:
    """
    Returns a function that ramps from start_value to end_value between ramp_start_step and ramp_end_step.
    """
    def ramp_annealing(global_step : int, iterations : int) -> float:
        if global_step < ramp_start_step:
            return start_value
        elif global_step > ramp_end_step:
            return end_value
        else:
            progress = (global_step - ramp_start_step) / (ramp_end_step - ramp_start_step)
            return start_value + progress * (end_value - start_value)
    return ramp_annealing

annealings = {
    "constant": get_constant_annealing,
    "ramp": get_ramp_annealing
}

class QNetwork(nn.Module):
    def __init__(self,
                 action_size : int,
                 q_network_arch : List[int],
                 observation_size : int,
                 torch_device : Union[str,th.device] = "cuda",
                 nets_num : int = 1,
                 initial_scale = 0.003,
                 use_weightnorm : bool = True):
        super().__init__()
        self._nets_num = nets_num
        self._obs_size = observation_size
        self._use_weightnorm = use_weightnorm
        self._q_nets = build_mlp_net(arch=q_network_arch,
                                     input_size=action_size + observation_size,
                                     output_size=1,
                                     ensemble_size=self._nets_num,
                                     return_ensemble_mean=False,
                                     use_weightnorm=self._use_weightnorm,
                                     use_torchscript=False,
                                     use_jit_fork=False,
                                     last_layer_init_func= lambda m: scale_layer_weights(m,initial_scale)).to(device=torch_device)
    
    # @th.compile(mode=compile_mode, fullgraph=True)
    def get_min_qval(self, observations, actions):
        qvals = self(observations, actions)
        # ggLog.info(f"qvals.size() = {qvals.size()}")
        # min_q = qvals[:,0]
        min_q = th.amin(qvals,dim=1)
        # min_q = min_q.squeeze(1)
        # ggLog.info(f"min_q.size() = {min_q.size()}")
        return min_q
    
    # @th.compile(mode=compile_mode, fullgraph=True)    
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
                        action_mean_init = 0.0,
                        use_weightnorm : bool = True):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self.device = torch_device
        self._obs_size = observation_size
        self._use_weightnorm = use_weightnorm
        # save action scaling factors as non-trained parameters
        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        self.action_bias : th.Tensor
        self.action_scale : th.Tensor
        self.register_buffer("action_scale", th.as_tensor((action_max - action_min) / 2.0, dtype=th.float32, device=torch_device))
        self.register_buffer("action_bias",  th.as_tensor((action_max + action_min) / 2.0, dtype=th.float32, device=torch_device))
        inner_activations = th.nn.LeakyReLU
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.act_fc = build_mlp_net(arch=policy_arch[:-1],input_size=observation_size, output_size=policy_arch[-1],
                                    last_activation_class=inner_activations,
                                    hidden_activations=inner_activations,
                                    use_weightnorm=self._use_weightnorm).to(device=torch_device)
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

    def forward(self, observation_batch):
        hidden_batch = self.act_fc(observation_batch)
        # dbg_check_finite(hidden_batch)
        mean = self.act_fc_mean(hidden_batch)
        log_std = self.act_fc_logstd(hidden_batch)
        log_std = (th.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min # clamp the log_std network output
        return mean, log_std

    @th_compile_ext(mode=compile_mode, fullgraph=True, copy_outs=True)
    def sample_action(self, observation_batch, reference_action : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        mean, log_std = self(observation_batch)
        if reference_action is not None:
            mean = mean + reference_action
        std = log_std.exp()
        x_t = mean + th.empty_like(mean).normal_(mean=0.0, std=1.0)*std # rsample has issues with torch.compile
        normal = th.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t) # get the probability of the actions that we sampled
        y_t = th.tanh(x_t) # squash the action in [-1,1]

        # scale mean and action to the proper bounds
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = log_prob - th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # correct the probability for the squashing and scaling
        log_prob = log_prob.sum(1, keepdim=True) # get probability per each multidimensional action, not for each action component

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
        gamma : float
        log_std_init : float
        max_grad_norm : float
        observation_space : gym.spaces.Space
        policy_arch : List[int]
        policy_lr : float
        policy_update_freq : int
        q_lr : float
        q_network_arch : List[int]
        target_entropy_annealing : tuple[Literal["constant", "ramp"], list[float | th.Tensor]] | None
        target_entropy_factor : float
        target_tau : float
        targets_update_freq : int
        torch_device : th.device
        critic_weight_decay : float
        actor_weight_decay : float

    def __init__(self,
                 action_size : int,
                 init_hparams : SAC_init_hparams,
                 observation_space : gym.spaces.Space,
                 action_init : th.Tensor | float = 0.0,
                 action_max : Union[float, List[float]] = 1.0,
                 action_min : Union[float, List[float]] = -1.0,
                 actor_feature_extractor : FeatureExtractor | None = None,
                 critic_feature_extractor : FeatureExtractor | None = None
                 ):
        super().__init__()
        self._init_args = get_func_input_args(exclude=[ "self",
                                                        "values",
                                                        "__class__",
                                                        "critic_feature_extractor",
                                                        "actor_feature_extractor"])
        ggLog.info(f"self._init_args = \n"+pprint.pformat(self._init_args))
        self._init_args = copy.deepcopy(self._init_args)
        init_hparams = copy.deepcopy(init_hparams)
        if init_hparams.target_entropy_factor is None:
            init_hparams.target_entropy_factor = -1.0
        if init_hparams.actor_observation_filter != None:
            if isinstance(observation_space, spaces.gym_spaces.Dict):
                actor_observation_space = spaces.gym_spaces.Dict({k:v for k,v in observation_space.spaces.items() if k in init_hparams.actor_observation_filter})
            else:
                raise RuntimeError(f"observation space must be a Dict to use actor_observation_filter, but it's a {type(observation_space)}")
        else:
            actor_observation_space = observation_space
        if init_hparams.critic_observation_filter != None:
            if isinstance(observation_space, spaces.gym_spaces.Dict):
                critic_observation_space = spaces.gym_spaces.Dict({k:v for k,v in observation_space.spaces.items() if k in init_hparams.critic_observation_filter})
            else:
                raise RuntimeError(f"observation space must be a Dict to use actor_observation_filter, but it's a {type(observation_space)}")
        else:
            critic_observation_space = observation_space
        if init_hparams.action_reference_obs_key is not None:
            action_init = 0.0
        self._hp = SAC.Hyperparams(q_lr=init_hparams.q_lr,
                                   policy_lr = init_hparams.policy_lr,
                                   gamma=init_hparams.gamma,
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
                                   torch_device = th.device(init_hparams.torch_device),
                                   target_entropy_factor = init_hparams.target_entropy_factor,
                                   observation_space = observation_space,
                                   feature_extractor_lr = init_hparams.feature_extractor_lr,
                                   batch_size = init_hparams.batch_size,
                                   max_grad_norm=init_hparams.max_grad_norm,
                                   log_std_init = init_hparams.actor_log_std_init,
                                   actor_observation_space = actor_observation_space,
                                   critic_observation_space = critic_observation_space,
                                   actor_observation_filter = init_hparams.actor_observation_filter,
                                   critic_observation_filter = init_hparams.critic_observation_filter,
                                   target_entropy_annealing = init_hparams.target_entropy_factor_annealing,
                                   action_reference_obs_key = init_hparams.action_reference_obs_key,
                                   critic_weight_decay = init_hparams.critic_weight_decay,
                                   actor_weight_decay = init_hparams.actor_weight_decay)
        self._obs_space_sizes = sizetree_from_space(observation_space)
        self.device = self._hp.torch_device
        self._critic_updates = 0
        self._alpha_updates = 0
        self._actor_updates = 0
        self._agent_updates = 0
        self._share_actor_critic_feature_extractor = (actor_feature_extractor==critic_feature_extractor and
                                                      init_hparams.actor_observation_filter==init_hparams.critic_observation_filter)
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
                                q_network_arch=init_hparams.q_network_arch,
                                torch_device=self._hp.torch_device,
                                nets_num=2)
        self._q_net_target = QNetwork(  observation_size=self._critic_feature_extractor.encoding_size(),
                                        action_size=self._hp.action_size,
                                        q_network_arch=init_hparams.q_network_arch,
                                        torch_device=self._hp.torch_device,
                                        nets_num=2)
        self._q_net_target.load_state_dict(self._q_net.state_dict())
        self._q_optimizer = optim.Adam(split_params_for_weight_decay(self._q_net, self._hp.critic_weight_decay), lr=self._hp.q_lr)
        self._actor = Actor(policy_arch=init_hparams.policy_arch,
                            observation_size=self._actor_feature_extractor.encoding_size(),
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.torch_device,
                            log_std_init=self._hp.log_std_init,
                            action_mean_init=self._hp.action_init)
        # self._actor_optimizer = optim.Adam(split_params_for_weight_decay(self._actor,self._hp.actor_weight_decay),
        #                                    lr=self._hp.policy_lr)
        self._base_target_entropy_factor = th.as_tensor(self._hp.target_entropy_factor, device=self._hp.torch_device, dtype=th.float32)
        self._target_entropy = self._base_target_entropy_factor*self._hp.action_size
        if init_hparams.target_entropy_factor_annealing is None:
            init_hparams.target_entropy_factor_annealing = ("constant", [self._base_target_entropy_factor])
        self._target_entropy_factor_annealing : AnnealingFunction = annealings[init_hparams.target_entropy_factor_annealing[0]](*init_hparams.target_entropy_factor_annealing[1])
        if self._hp.auto_entropy_temperature:
            self._log_alpha = th.zeros(1, requires_grad=True, device=init_hparams.torch_device)
            self._alpha = self._log_alpha.exp().detach()
            # self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = th.as_tensor(constant_entropy_temperature).to(device=self._hp.torch_device, non_blocking=self._hp.torch_device.type=="cuda")
            self._log_alpha = self._alpha.log().detach()

        self._actor_and_alpha_optimizer = optim.Adam([{ "params":[self._log_alpha], "lr":self._hp.q_lr}]+
                                                      split_params_for_weight_decay(self._actor,self._hp.actor_weight_decay,
                                                                                    extra_kwargs={"lr":self._hp.policy_lr}))

        if self._hp.feature_extractor_lr > 0:
            critic_extractor_params = list(self._critic_feature_extractor.parameters())
            if len(critic_extractor_params) > 0:
                self._critic_feature_extractor_optimizer = optim.Adam(critic_extractor_params, lr=self._hp.feature_extractor_lr)
            else:
                self._critic_feature_extractor_optimizer = None

            if self._share_actor_critic_feature_extractor:
                self._actor_feature_extractor_optimizer = self._critic_feature_extractor_optimizer
            else:
                actor_extractor_params = list(self._actor_feature_extractor.parameters())
                if len(actor_extractor_params) > 0:
                    self._actor_feature_extractor_optimizer = optim.Adam(actor_extractor_params, lr=self._hp.feature_extractor_lr)
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
        # This was optimized by improving GPU usage via cudagraphs, profiling with Nsight Systems
        # The profiling command was:
        #  sudo nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop \
        #    --cudabacktrace=true -x true --force-overwrite true -o my_profile -s cpu \
        #    virtualenv/lrjax/bin/python3 src/rreal/src/rreal/examples/half_cheetah.py --comment t --algo sac
        # Then the produced file can be drag and dropped into Nsight Systems GUI to see the profiling results (e.g. the timeline)        
        # Can still be optimized more, but some segments are tricky to include in th.compile and behave weird
        
        self._stats = { "tot_grad_steps_count":0,
                        "q_loss":0.0,
                        "actor_loss":0.0,
                        "alpha_loss":0.0,
                        "val_q_loss":0.0,
                        "val_actor_loss":0.0,
                        "val_alpha_loss":0.0,
                        "alpha":0.0}
        
    def _nvtx_startup(self):
        if self._enable_nvtx and self._agent_updates == 10:
            th.cuda.cudart().cudaProfilerStart()

    def _nvtx_stop(self):
        if self._enable_nvtx and self._agent_updates > 13:
            th.cuda.cudart().cudaProfilerStop()  

    def _mark_nvtx(self, name : str):
        if self._enable_nvtx and self._agent_updates >10:
            th.cuda.nvtx.mark(name)

    def _nvtx_start_range(self, name : str):
        if self._enable_nvtx and self._agent_updates >10:
            th.cuda.nvtx.range_push(name)

    def _nvtx_end_range(self):
        if self._enable_nvtx and self._agent_updates >10:
            th.cuda.nvtx.range_pop()

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
        flat_self = flatten_tensor_tree(self._init_args)
        flat_load = flatten_tensor_tree(extra["init_args"])
        equal_fields = map2_tensor_tree(flat_self, flat_load, lambda a,b: th.all(a==b) if isinstance(a,th.Tensor) else a==b)
        if not all(equal_fields):
            ggLog.warn("init args of loaded model differ from those of self.")
            load_yaml_args = yaml.dump(extra['init_args'])
            original_yaml_args = yaml.dump(self._init_args)
            ggLog.warn(f"self._init_args = \n{original_yaml_args}")
            ggLog.warn(f"load init_args  = \n{load_yaml_args}")
            differing_fields = [k for k,v in equal_fields.items() if v==False]
            ggLog.warn(f"Differing fields:")
            for k in differing_fields:
                ggLog.warn(f"k:\n"
                           f"    self={flat_self[k]}\n"
                           f"    load={flat_load[k]}")
            # diffs = ndiff(   original_yaml_args.splitlines(keepends=True),
            #                 load_yaml_args.splitlines(keepends=True))
            # diffs = [l for l in diffs if len(l)>0 and l[0] != ' ']
            # ggLog.warn(f"Args comparison with loaded model:\n{''.join(diffs)}")
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
    def predict_action(self, observation_batch, deterministic = False, info_return : dict | None = None):
        th.compiler.cudagraph_mark_step_begin()
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
        observation_batch_enc = self._actor_feature_extractor.extract_features(observation_batch)
        reference_action : th.Tensor = observation_batch[self._hp.action_reference_obs_key] if self._hp.action_reference_obs_key is not None else None
        action, log_prob, mean, log_std = self._actor.sample_action(observation_batch_enc, reference_action=reference_action)
        if not is_batched:
            action = action.squeeze()
            mean = mean.squeeze()
            log_prob = log_prob.squeeze()
        if info_return is not None:
            info_return["mean"] = mean
            info_return["log_prob"] = log_prob
            info_return["action"] = action
            info_return["log_std"] = log_std
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

    @th.compile(mode=compile_mode, fullgraph=True)
    def _compute_critic_loss(self, transitions : TransitionBatch):
        critic_obss = self.get_critic_subobservation(transitions.observations)
        critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss)
        with th.no_grad():
            critic_next_obss = self.get_critic_subobservation(transitions.next_observations)
            crit_next_enc_obss = self._critic_feature_extractor.extract_features(critic_next_obss)   
            if self._share_actor_critic_feature_extractor:     
                actor_next_obss = critic_next_obss
                actor_next_obss_enc = crit_next_enc_obss
            else:
                actor_next_obss = self.get_actor_subobservation(transitions.next_observations)
                actor_next_obss_enc = self._actor_feature_extractor.extract_features(actor_next_obss)
            reference_action : th.Tensor = actor_next_obss[self._hp.action_reference_obs_key] if self._hp.action_reference_obs_key is not None else None
            
            # Compute next-values for TD
            next_state_actions, next_state_log_pi, _, _ = self._actor.sample_action(actor_next_obss_enc, reference_action = reference_action)
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

    @th.compile(mode=compile_mode, fullgraph=False)
    def _critic_opt_step(self, q_loss : th.Tensor):
        simplified_clip_grad_norm_(list(self._q_net.parameters()), self._hp.max_grad_norm)
        self._q_optimizer.step()
        self._last_q_loss.copy_(q_loss.detach())

    def _update_critic(self, transitions : TransitionBatch):
        # ggLog.info(f"critic update...")
        
        # self._nvtx_start_range("_update_critic")
        self._q_optimizer.zero_grad(set_to_none=True)
        # self._nvtx_start_range("critic forward")
        # ggLog.info(f"compute_critic_loss...")
        q_loss = self._compute_critic_loss(transitions)
        # ggLog.info(f"compute_critic_loss done")
        # self._nvtx_end_range()
        # self._nvtx_start_range("critic backward")
        q_loss.backward()
        # self._nvtx_end_range()
        # self._nvtx_start_range("critic opt")
        with th.no_grad():
            self._critic_opt_step(q_loss)
        # self._nvtx_end_range()
        self._critic_updates += 1
        # self._nvtx_end_range()
        # ggLog.info(f"critic update done")


    # @th.compile(mode=compile_mode, fullgraph=True)
    def _compute_actor_loss(self, transitions : TransitionBatch, freeze_critic : bool = False):
        # self._mark_nvtx("_compute_actor_loss")
        actor_obss = self.get_actor_subobservation(transitions.observations)
        # self._mark_nvtx("actor_enc")
        actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss).clone()
        # self._mark_nvtx("get ref")
        reference_action : th.Tensor = actor_obss[self._hp.action_reference_obs_key] if self._hp.action_reference_obs_key is not None else None
        # self._mark_nvtx("sample")
        act, act_log_prob, _, _ = self._actor.sample_action(actor_enc_obss, reference_action=reference_action)
        act, act_log_prob = act.clone(), act_log_prob.clone() # prevent issues with cuda graphs
        # with th.no_grad():
        # self._mark_nvtx("crit_enc")
        if self._share_actor_critic_feature_extractor:
            critic_enc_obss = actor_enc_obss
        else:
            critic_obss = self.get_critic_subobservation(transitions.observations)
            critic_enc_obss = self._critic_feature_extractor.extract_features(critic_obss).clone()
        # self._mark_nvtx("get_q")
        if freeze_critic: # Needed if we are updating actor and critic together, if done separately we just ignore these grads at optimizer time
            self._q_net.requires_grad_(False)
        min_q_pi = self._q_net.get_min_qval(critic_enc_obss, act) # cannot reuse those from _update_value_func, the value function has changed
        if freeze_critic:
            self._q_net.requires_grad_(True)
        # ggLog.info(f"min_q_pi.size() = {min_q_pi.size()}")
        # ggLog.info(f"act_log_prob.size() = {act_log_prob.size()}")
        # self._mark_nvtx("ret_q")
        return ((self._alpha * act_log_prob) - min_q_pi).mean()
    
    # @th.compile(mode=compile_mode, fullgraph=True)
    def _alpha_loss(self, act_log_prob : th.Tensor):
        return (-self._log_alpha.exp() * (act_log_prob + self._target_entropy)).mean()
    
    # @th.compile(mode=compile_mode, fullgraph=True)
    def _alpha_stats(self, act_log_prob : th.Tensor):
        return (act_log_prob.mean(),
                act_log_prob.min(),
                act_log_prob.max(),
                act_log_prob.quantile(0.95),
                act_log_prob.quantile(0.05) )

    @th.compile(mode=compile_mode, fullgraph=True)
    def _compute_alpha_loss(self, transitions : TransitionBatch):
        with th.no_grad():
            actor_obss = self.get_actor_subobservation(transitions.observations)
            actor_enc_obss = self._actor_feature_extractor.extract_features(actor_obss)
            reference_action : th.Tensor = actor_obss[self._hp.action_reference_obs_key] if self._hp.action_reference_obs_key is not None else None
            _, act_log_prob, _, _ = self._actor.sample_action(actor_enc_obss, reference_action=reference_action)
            stats = self._alpha_stats(act_log_prob)
        return self._alpha_loss(act_log_prob), stats
    

    @th.compile(mode=compile_mode, fullgraph=True)
    def _compute_actor_and_alpha_loss(self, transitions : TransitionBatch):
        actor_loss = self._compute_actor_loss(transitions)
        # self._start_range_nvtx("actor opt")
        
        # self._start_range_nvtx("_update_alpha")
        if self._hp.auto_entropy_temperature:
            alpha_loss, alpha_stats = self._compute_alpha_loss(transitions)
            loss = actor_loss + alpha_loss
        else:
            alpha_loss, alpha_stats = None, None
            loss = actor_loss
        
        return loss, actor_loss, alpha_loss, alpha_stats
    
    @th.compile(mode=compile_mode, fullgraph=True)
    def _compute_all_losses(self, transitions):
        q_loss = self._compute_critic_loss(transitions)
        actor_loss = self._compute_actor_loss(transitions, freeze_critic=True)
        # self._start_range_nvtx("actor opt")
        
        # self._start_range_nvtx("_update_alpha")
        if self._hp.auto_entropy_temperature:
            alpha_loss, alpha_stats = self._compute_alpha_loss(transitions)
            loss = actor_loss + alpha_loss + q_loss
        else:
            alpha_loss, alpha_stats = None, None
            loss = actor_loss + q_loss
        return loss, q_loss, actor_loss, alpha_loss, alpha_stats
    
    @th.compile(mode=compile_mode, fullgraph=False)
    def _actor_and_alpha_opt_step(self, actor_loss : th.Tensor, alpha_loss : th.Tensor | None):
        simplified_clip_grad_norm_(list(self._actor.parameters()), self._hp.max_grad_norm)
        simplified_clip_grad_norm_([self._log_alpha], self._hp.max_grad_norm)
        self._actor_and_alpha_optimizer.step()
        self._alpha.fill_(self._log_alpha.exp().detach().view(tuple())) # keep the same address to make cudagraphs happy
        self._last_actor_loss.copy_(actor_loss.detach())
        if alpha_loss is not None:
            self._last_alpha_loss.copy_(alpha_loss.detach())

    def _update_actor_and_alpha(self, transitions : TransitionBatch) -> tuple[th.Tensor, th.Tensor | None]:
        # We aggregate actor and alpha to join the two compilation regions and cuda graphs, so to reduce overhead
        # self._nvtx_start_range("_update_actor_and_alpha")
        self._actor_and_alpha_optimizer.zero_grad(set_to_none=True)
        # self._nvtx_start_range("_compute_actor_and_alpha_loss")
        actor_alpha_loss, actor_loss, alpha_loss, alpha_stats = self._compute_actor_and_alpha_loss(transitions)
        # self._nvtx_end_range()
        # self._nvtx_start_range("actor_alpha backward")
        actor_alpha_loss.backward()
        # self._nvtx_end_range()
        # self._nvtx_start_range("actor_alpha opt")
        with th.no_grad():
            self._actor_and_alpha_opt_step(actor_loss, alpha_loss)
        # self._nvtx_end_range()

        if alpha_stats is not None:
            self._stats.update({k:v for k,v in zip(["avg_log_prob",
                                                    "min_log_prob",
                                                    "max_log_prob",
                                                    "q95_log_prob",
                                                    "q05_log_prob"],alpha_stats)})            
        self._alpha_updates += 1
        self._actor_updates += 1
        # self._nvtx_end_range()

    def _update_all(self, transitions):
        # self._nvtx_start_range("_update_all")
        # raise NotImplementedError("This does not work for some reason")
        self._actor_and_alpha_optimizer.zero_grad(set_to_none=True)
        self._q_optimizer.zero_grad(set_to_none=True)

        # self._nvtx_start_range("_compute_all_losses")
        loss, q_loss, actor_loss, alpha_loss, alpha_stats = self._compute_all_losses(transitions)
        # self._nvtx_end_range()
        # self._nvtx_start_range("all backward")
        loss.backward()
        # self._nvtx_end_range()

        # self._nvtx_start_range("all opt")
        with th.no_grad():
            #TODO: merge the optimizers?
            self._critic_opt_step(q_loss)
            self._actor_and_alpha_opt_step(actor_loss, alpha_loss)
        # self._nvtx_end_range()

        if alpha_stats is not None:
            self._stats.update({k:v for k,v in zip(["avg_log_prob",
                                                    "min_log_prob",
                                                    "max_log_prob",
                                                    "q95_log_prob",
                                                    "q05_log_prob"],alpha_stats)})            
        self._critic_updates += 1
        self._alpha_updates += 1
        self._actor_updates += 1
        # self._nvtx_end_range()
        
    @staticmethod
    def _target_update(param, target_param, tau):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @th.compile(mode=compile_mode, fullgraph=True)        
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
        th.compiler.cudagraph_mark_step_begin()
        # self._nvtx_startup()
        # self._nvtx_start_range(f"iteration{self._critic_updates}")
        if self._enable_feature_extractor_training:
            if self._critic_feature_extractor_optimizer is not None:
                self._critic_feature_extractor_optimizer.zero_grad(set_to_none=True)
            if self._actor_feature_extractor_optimizer is not None:
                self._actor_feature_extractor_optimizer.zero_grad(set_to_none=True)
        update_actor_and_alpha = self._critic_updates % self._hp.policy_update_freq == 0
        if update_actor_and_alpha:
            self._update_all(transitions)
            for _ in range(self._hp.policy_update_freq-1): # do the remaining updates
                self._update_actor_and_alpha(transitions=transitions)
        else:
            self._update_critic(transitions)

        if self._critic_updates % self._hp.targets_update_freq == 0:
            # self._nvtx_start_range("_update_target_nets")
            self._update_target_nets()
            # self._nvtx_end_range()
        if self._enable_feature_extractor_training:
            self._update_feature_extractor()
        self._agent_updates += 1
        # self._nvtx_end_range()
        # self._nvtx_stop()
        return self._last_q_loss, self._last_actor_loss, self._last_alpha_loss

    # def _update(self, transitions : TransitionBatch):
    #     # ggLog.info(f" ----------- SAC update {self._agent_updates}...")
    #     th.compiler.cudagraph_mark_step_begin()
    #     # sync_dbg_mode = th.cuda.get_sync_debug_mode()
    #     # th.cuda.set_sync_debug_mode("error")
    #     # Mark the beginning of cuda graphs to help the compile.Docs say "CUDA Graphs will free tensors of
    #     #  a prior iteration. A new iteration is started on each invocation of torch.compile, so long as 
    #     # there is not a pending backward that has not been called.". Not sure what it means, but marking these
    #     # should be helpful
    #     # th.compiler.cudagraph_mark_step_begin()
    #     # self._nvtx_startup()
    #     # self._nvtx_start_range(f"iteration{self._critic_updates}")

    #     if self._enable_feature_extractor_training:
    #         if self._critic_feature_extractor_optimizer is not None:
    #             self._critic_feature_extractor_optimizer.zero_grad(set_to_none=True)
    #         if self._actor_feature_extractor_optimizer is not None:
    #             self._actor_feature_extractor_optimizer.zero_grad(set_to_none=True)

    #     self._update_critic(transitions = transitions)
    #     if self._critic_updates % self._hp.policy_update_freq == 0:
    #         for _ in range(self._hp.policy_update_freq):
    #             self._update_actor_and_alpha(transitions=transitions) # TODO: is it good to update twice with the same batch
    #     if self._critic_updates % self._hp.targets_update_freq == 0:
    #         # self._nvtx_start_range("_update_target_nets")
    #         self._update_target_nets()
    #         # self._nvtx_end_range()
    #     if self._enable_feature_extractor_training:
    #         self._update_feature_extractor()
    #     # self._nvtx_end_range()
    #     self._agent_updates += 1
    #     # self._nvtx_stop()      
    #     # th.cuda.set_sync_debug_mode(sync_dbg_mode)
    #     # ggLog.info(f"sac update done")
    #     return self._last_q_loss, self._last_actor_loss, self._last_alpha_loss
    
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
        # ggLog.info(f":::::::::::::::::::::::: train_model: global_step={global_step}")
        q_act_alpha_losses = [None]*iterations
        target_entropy_cpu = self._target_entropy_factor_annealing(global_step, iterations)*self._hp.action_size
        self._target_entropy.copy_(th.as_tensor(target_entropy_cpu).to(device=self.device, dtype=th.float32, non_blocking=self.device.type=="cuda"))
        for i in range(iterations):
            transitions = buffer.sample(self._hp.batch_size)
            transitions = map_tensor_tree(transitions, lambda t : t.to(device=self.device, non_blocking=self.device.type=="cuda"))
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
                            "alpha":self._alpha.clone(),
                            "target_entropy":target_entropy_cpu})
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
        grad_steps_done = 0
        if global_step > learning_start_step:
            iterations = grad_steps if grad_steps!="auto" else 10
            while (grad_steps != "auto" and not trained) or (grad_steps == "auto" and collector.is_collecting()):
                trained = True
                q_loss, actor_loss, alpha_loss = model.train_model(global_step, iterations, buffer)
                grad_steps_done += iterations
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
        t_after_wait = time.monotonic()
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
        global_step += steps_to_collect
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
        t = time.monotonic()
        # ggLog.info(f"global_steps = {global_step}")
        if global_step - last_log_steps > log_freq_vstep*num_envs:
            last_log_steps = global_step
            log_async(f"SAC: expsteps={global_step} q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}",
                      tensors=dict(q_loss=q_loss,actor_loss=actor_loss,alpha_loss=alpha_loss))
            # ggLog.info(f"SAC: expsteps={global_step} q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}")
            ggLog.info(f"OFFTRAIN: expstps:{global_step}"
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
                       f" ips={grad_steps_done_sl/t_train_sl:.2f}"
                       f" alltime_fps={global_step/(t-start_time):.2f} alltime_ips={model._tot_grad_steps_count/(t-start_time):.2f}")
            dictlist = [f"{k}:{v:.6g}" for k,v in collector.get_stats().items()]
            ggLog.info(f"Collection: {', '.join(dictlist)}")
            t_train_sl, t_coll_sl, t_tot_sl, steps_sl, t_val_sl, t_buff_sl, t_add_sl, t_start_sl, t_end_callbacks_sl, t_wait_collect_sl, grad_steps_done_sl = 0,0,0,0,0,0,0,0,0,0,0
            free, total = th.cuda.mem_get_info(th.device('cuda:0'))
            mem_used_MB = (total - free) / 1024 ** 2
            # ggLog.info(f"{t}: cuda mem usage = {mem_used_MB}")
        # jax.profiler.save_device_memory_profile(f"jax_memory_{t}.prof")
        # th.cuda.memory._dump_snapshot(f"memory_{t}_th.pickle")
        adarl.utils.sigint_handler.haltOnSigintReceived()
    callbacks.on_training_end()

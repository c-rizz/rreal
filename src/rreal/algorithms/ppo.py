# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import copy
from rreal.algorithms.rl_agent import RLAgent
from adarl.utils.tensor_trees import sizetree_from_space, map_tensor_tree
from adarl.utils.utils import numpy_to_torch_dtype_dict
from rreal.utils import build_mlp_net
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor

from rreal.algorithms.sac_helpers import EnvBuilderProtocol, VecEnvBuilderProtocol, build_vec_env, wrap_with_logger, build_eval_callbacks
from adarl.utils.callbacks import CheckpointCallbackRB, TrainingCallback, CallbackList
import adarl.utils.session
import inspect
import wandb
from typing import Any
import adarl.utils.dbg.ggLog as ggLog
from typing_extensions import override
import adarl.utils.sigint_handler
from adarl.utils.wandb_wrapper import wandb_log

# def make_env(env_id, idx, capture_video, run_name, gamma):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env = gym.wrappers.ClipAction(env)
#         env = gym.wrappers.NormalizeObservation(env)
#         env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
#         env = gym.wrappers.NormalizeReward(env, gamma=gamma)
#         env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
#         return env

#     return thunk


class PPORolloutBuffer():
    def __init__(self,  num_steps : int, num_envs : int, storage_torch_device : th.device,
                        action_size : int,
                        observation_space : gym.spaces.Dict):
        self._num_envs = num_envs
        self._num_steps = num_steps
        self._storage_th_device = storage_torch_device
        print(f"observation_space ={observation_space}")
        self._start_observations = {
            key: th.zeros(  size=(num_steps+1, num_envs) + space.shape,
                            dtype=numpy_to_torch_dtype_dict[space.dtype],
                            device = self._storage_th_device)
            for key, space in observation_space.spaces.items()
        }
        self._actions = th.zeros((num_steps+1, num_envs, action_size), device=self._storage_th_device)
        self._act_logprobs = th.zeros((num_steps+1, num_envs), device=self._storage_th_device)
        self._obs_values = th.zeros((num_steps+1, num_envs), device=self._storage_th_device)
        self._rewards = th.zeros((num_steps, num_envs), device=self._storage_th_device)
        self._terminated = th.zeros((num_steps, num_envs), device=self._storage_th_device)
        self._truncated = th.zeros((num_steps, num_envs), device=self._storage_th_device)
        self._pos = 0

    def add(self, consequent_obss, actions, logprobs, rewards, terminateds, truncateds, values):
        for k in consequent_obss.keys():
            self._start_observations[k][self._pos+1] = consequent_obss[k]
        self._actions[self._pos] = actions
        self._rewards[self._pos] = rewards
        self._terminated[self._pos] = terminateds
        self._truncated[self._pos] = truncateds
        self._obs_values[self._pos] = values
        self._act_logprobs[self._pos] = logprobs
        self._pos += 1

    def set_logprobs_values(self, action, logprobs, values):
        self._actions[self._pos] = action
        self._obs_values[self._pos] = values
        self._act_logprobs[self._pos] = logprobs


    def add_start_obss(self, start_obss):
        if self._pos != 0:
            raise RuntimeError()
        for k in start_obss.keys():
            self._start_observations[k][0] = start_obss[k]


    def reset(self):
        self._pos = 0

    def get_rollout_data(self):
        return ({k:obs[:self._pos+1] for k,obs in self._start_observations.items()},
                self._actions[:self._pos],
                self._rewards[:self._pos],
                self._terminated[:self._pos],
                self._truncated[:self._pos],
                self._obs_values[:self._pos+1],
                self._act_logprobs[:self._pos+1])

    


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.critic = nn.Sequential(
#             _layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             _layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             _layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             _layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             _layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             _layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
#         )
#         self.actor_logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))

#     def get_value(self, x):
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         action_mean = self.actor_mean(x)
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         action_std = th.exp(action_logstd)
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

class Critic(nn.Module):
    def __init__(self,
                 q_network_arch : list[int],
                 observation_size : int,
                 torch_device : str | th.device = "cuda"):
        super().__init__()
        self._val_net = build_mlp_net(arch=q_network_arch,
                                     input_size=observation_size,
                                     output_size=1,
                                     use_weightnorm=True,
                                     use_torchscript=True).to(device=torch_device)
    def get_value(self, x) -> th.Tensor:
        return self._val_net(x)


class Actor(nn.Module):
    def __init__(self,  action_size,
                        observation_size : int,
                        policy_arch = [256,256],
                        action_max : float | th.Tensor = 1,
                        action_min : float | th.Tensor = -1,
                        log_std_max = 2,
                        log_std_min = -5,
                        torch_device : str | th.device = "cuda"):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._act_mean = build_mlp_net(arch=policy_arch,
                                     input_size=observation_size,
                                     output_size=action_size,
                                     use_weightnorm=True,
                                     use_torchscript=True).to(device=torch_device)
        self.actor_logstd = nn.Parameter(th.zeros(1, action_size, device=torch_device))
        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        # save action scaling factors as non-trained parameters
        self.register_buffer("action_scale", th.as_tensor((action_max - action_min) / 2.0, dtype=th.float32, device=torch_device))
        self.register_buffer("action_bias",  th.as_tensor((action_max + action_min) / 2.0, dtype=th.float32, device=torch_device))

    def forward(self, observation_batch):
        mean = self._act_mean(observation_batch)
        log_std = self.actor_logstd.expand_as(mean)
        log_std = (th.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min # clamp the log_std network output
        return mean, log_std
    

    def get_act_logprob_mean_entropy(self, observation_batch, action : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # return self._get_action_noscale(observation_batch, action)
        mean, log_std = self(observation_batch)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        if action is None:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        else:
            x_t = (action - self.action_bias)/self.action_scale
        y_t = th.tanh(x_t) # squash the action in [-1,1]
        log_prob = normal.log_prob(x_t) # get the probability of the actions that we sampled

        # scale mean and action to the proper bounds
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = log_prob - th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # correct the probability for the squashing
        log_prob = log_prob.sum(1, keepdim=True) # get probability per each multidimensional action, not for each action component

        return action, log_prob, mean, normal.entropy().sum(1)
    

    def _get_action_noscale(self, x, action=None):
        action_mean = self._act_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), action_mean, probs.entropy().sum(1)

class PPO(RLAgent):
    @dataclass
    class Hyperparams:
        minibatch_size: int
        th_device : th.device
        action_len : int
        observation_space : gym.spaces.Space
        policy_arch : list[int]
        q_network_arch : list[int]
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
        clip_coef: float = 0.2
        """the surrogate clipping coefficient"""
        clip_vloss: bool = True
        """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
        ent_coef: float = 0.0
        """coefficient of the entropy"""
        vf_coef: float = 0.5
        """coefficient of the value function"""
        max_grad_norm: float = 0.5
        """the maximum norm for the gradient clipping"""
        target_kl: float = None
        """the target KL divergence threshold"""


    def __init__(self,  hyperparams : Hyperparams,
                        feature_extractor : FeatureExtractor | None = None,):
        super().__init__()
        self._hp = copy.deepcopy(hyperparams)
        if feature_extractor is None:
            self._feature_extractor = StackVectorsFeatureExtractor(observation_space=self._hp.observation_space)
        else:
            raise NotImplementedError()
            self._feature_extractor = feature_extractor
        self._actor = Actor(action_size=self._hp.action_len,
                            observation_size=self._feature_extractor.encoding_size(),
                            policy_arch=self._hp.policy_arch,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.th_device)
        self._critic = Critic(  observation_size=self._feature_extractor.encoding_size(),
                                q_network_arch=self._hp.q_network_arch,
                                torch_device=self._hp.th_device)
        self._q_optimizer = optim.Adam(self._critic.parameters(), lr=self._hp.q_lr)
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp.policy_lr)
        self._grad_steps_count = 0
        self._epochs_count = 0

    def train_model(self, buffer : PPORolloutBuffer):
        raw_obss, acts, rews, terms, truncs, vals, logprobs = buffer.get_rollout_data()
        enc_obss = self._feature_extractor.extract_features(raw_obss) 
        # obss and vals and are aligned, acts, logprobs, rews, terms and trunct are their consequence.
        # So each transition at time t (s,a,s',r,term,trunc) is contained in:
        #   obss[t], act[t], obss[t+1], rew[t], term[t], trunc[t]
        # In cleanRL's original implementation:
        #   obss[t], act[t], obss[t+1], rew[t], done[t+1]
        num_steps = acts.shape[1]
        # bootstrap value if not done
        advantages = th.zeros_like(rews, device=self._hp.th_device)
        with th.no_grad():
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = th.logical_not(th.logical_or(terms[t], truncs[t]))
                delta = rews[t] + self._hp.gamma * vals[t + 1] * nextnonterminal - vals[t]
                lastgaelam = delta + self._hp.gamma * self._hp.gae_lambda * nextnonterminal * lastgaelam # this line unrolls the delta as in eq. 12 of Schulman 2017
                advantages[t] = lastgaelam
            returns = advantages + vals[:-1]

        # flatten the batch
        b_obs = map_tensor_tree(enc_obss, lambda t: t[:-1].reshape(shape=(-1,)+ t.size()[2:]))
        b_logprobs = logprobs.reshape(-1)
        b_actions = acts.reshape(-1,self._hp.action_len)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = vals[:-1].reshape(-1)

        batch_size = self._hp.num_envs*self._hp.num_steps
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        # clipfracs = []
        for epoch in range(self._hp.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self._hp.minibatch_size):
                end = start + self._hp.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = map_tensor_tree(b_obs, lambda t: t[mb_inds])
                _, newlogprob, _,  entropy = self._actor.get_act_logprob_mean_entropy(mb_obs, b_actions[mb_inds])
                newvalue = self._critic.get_value(mb_obs)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # clipfracs += [((ratio - 1.0).abs() > self._hp.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self._hp.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self._hp.clip_coef, 1 + self._hp.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._hp.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self._hp.clip_coef,
                        self._hp.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self._hp.ent_coef * entropy_loss + v_loss * self._hp.vf_coef

                self._q_optimizer.zero_grad(set_to_none=True)
                self._actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._actor.parameters(), self._hp.max_grad_norm)
                nn.utils.clip_grad_norm_(self._critic.parameters(), self._hp.max_grad_norm)
                self._q_optimizer.step()
                self._actor_optimizer.step()
                self._grad_steps_count += 1
            self._epochs_count += 1

            if self._hp.target_kl is not None and approx_kl > self._hp.target_kl:
                break
            wandb_log({ "ppo/loss": loss,
                        "ppo/entropy_loss" : entropy_loss,
                        "ppo/pg_loss" : pg_loss,
                        "ppo/v_loss" : v_loss,
                        "ppo/grad_steps" : self._grad_steps_count,
                        "ppo/epochs" : self._epochs_count})

    
    @override
    def save(self, path : str):
        pass

    @override
    def load_(self, path : str):
        pass
    
    @override
    def load(cls, path : str):
        pass

    def get_values(self, observations):
        enc_obss = self._feature_extractor.extract_features(observations) 
        return self._critic.get_value(enc_obss)
    
    def get_act_logprob(self, observations):
        enc_obss = self._feature_extractor.extract_features(observations) 
        act, logprob, _, _ = self._actor.get_act_logprob_mean_entropy(enc_obss)
        return act, logprob

        
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
        if not isinstance(self._vec_env.unwrapped.single_action_space, gym.spaces.Box):
            raise NotImplementedError()
        self._single_action_space = self._vec_env.unwrapped.single_action_space
        if not isinstance(self._vec_env.unwrapped.single_observation_space, gym.spaces.Dict):
            raise NotImplementedError()
        self._single_observation_space = self._vec_env.unwrapped.single_observation_space
        self._device = th_device
        self._num_envs = self._vec_env.unwrapped.num_envs
        self._curr_obss, _ = self._vec_env.reset(seed=seed)
        self._curr_obss = map_tensor_tree(self._curr_obss, lambda t: th.as_tensor(t, device = th_device))


    def set_agent(self, agent : PPO):
        self._agent = agent

    def collect_rollout(self, buffer : PPORolloutBuffer, vsteps_to_collect : int, agent : PPO):
        with th.no_grad():
            terminated_eps = th.as_tensor(0, device=self._device)
            buffer.reset()
            buffer.add_start_obss(self._curr_obss)
            for step in range(0, vsteps_to_collect):
                # ALGO LOGIC: action logic
                actions, logprobs = self._agent.get_act_logprob(self._curr_obss)
                values = agent.get_values(self._curr_obss)

                next_obss, rewards, terminations, truncations, infos = self._vec_env.step(actions)
                rewards = th.as_tensor(rewards, device=self._device).view(-1)
                self._curr_obss = map_tensor_tree(next_obss, lambda t: th.as_tensor(t, device = self._device))
                buffer.add( consequent_obss=self._curr_obss,
                            actions=actions,
                            logprobs=logprobs.reshape(self._num_envs),
                            rewards=rewards.reshape(self._num_envs),
                            terminateds=terminations.reshape(self._num_envs),
                            truncateds=truncations.reshape(self._num_envs),
                            values=values.reshape(self._num_envs))
                terminated_eps += th.logical_or(truncations, terminations).count_nonzero()
            actions, logprobs = self._agent.get_act_logprob(self._curr_obss)
            values = agent.get_values(self._curr_obss)
            buffer.set_logprobs_values( actions.reshape(self._num_envs,-1), 
                                        logprobs.reshape(self._num_envs), 
                                        values.reshape(self._num_envs))
            return terminated_eps
    
    def observation_space(self):
        return self._single_observation_space
    
    def single_action_space(self):
        return self._single_action_space
    
    def num_envs(self):
        return self._vec_env.unwrapped.num_envs
    
    def close(self):
        pass

def train_on_policy(collector : Collector,
                    model : PPO,
                    num_steps : int,
                    storage_torch_device : th.device,
                    train_steps : int,
                    callbacks : list[TrainingCallback]):
    buffer = PPORolloutBuffer(num_envs=collector.num_envs(),
                              num_steps=num_steps,
                              storage_torch_device=storage_torch_device,
                              observation_space=collector.observation_space(),
                              action_size=np.prod(collector.single_action_space().shape))
    callback = CallbackList(callbacks)
    callback.on_training_start()
    collected_steps = 0
    ep_counter = 0 
    for iteration in range(train_steps):
        callback.on_collection_start()
        terminated_eps = collector.collect_rollout(buffer=buffer, vsteps_to_collect=num_steps, agent=model)
        collected_steps += num_steps*collector.num_envs()
        ep_counter += terminated_eps
        adarl.utils.session.default_session.run_info["collected_episodes"].value = ep_counter
        adarl.utils.session.default_session.run_info["collected_steps"].value = collected_steps
        callback.on_collection_end(collected_episodes=int(terminated_eps.item()),
                                   collected_steps=num_steps,
                                   collected_data=None)
        model.train_model(buffer)
        adarl.utils.sigint_handler.haltOnSigintReceived()
    callback.on_training_end()

@dataclass
class PPO_hyperparams():
    minibatch_size : int
    th_device : th.device
    policy_arch : list[int]
    q_network_arch : list[int]
    q_lr : float
    policy_lr : float
    update_epochs : int
    total_steps : int
    num_envs : int
    num_steps : int
    gamma : float

def ppo_train(  seed : int,
                folderName : str,
                run_id : str,
                args,
                env_builder : EnvBuilderProtocol | None,
                vec_env_builder : VecEnvBuilderProtocol | None,
                env_builder_args : dict,
                agent_hyperparams : PPO_hyperparams,
                max_episode_duration : int,
                validation_buffer_size : int,
                validation_holdout_ratio : float,
                validation_batch_size : int,
                eval_configurations : list[dict] = [],
                checkpoint_freq : int = 100,
                collector_device : th.device | None = None,
                debug_level : int = 2,
                no_wandb : bool = False):

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
    np.random.seed(seed)
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
    vec_env_builder = wrap_with_logger(vec_env_builder)
    # env setup
    collector = Collector(  vec_env_builder=vec_env_builder,
                            env_builder_args=env_builder_args,
                            th_device=agent_hyperparams.th_device,
                            run_folder=run_folder,
                            num_envs=agent_hyperparams.num_envs,
                            seed=seed)
    observation_space = collector.observation_space()
    action_space = collector.single_action_space()
    model = PPO(hyperparams=PPO.Hyperparams(minibatch_size=agent_hyperparams.minibatch_size,
                                            th_device=agent_hyperparams.th_device,
                                            action_len=np.prod(action_space.shape),
                                            observation_space=observation_space,
                                            policy_arch=agent_hyperparams.policy_arch,
                                            q_network_arch=agent_hyperparams.q_network_arch,
                                            action_max=th.as_tensor(action_space.high, device=agent_hyperparams.th_device),
                                            action_min=th.as_tensor(action_space.low, device=agent_hyperparams.th_device),
                                            q_lr=agent_hyperparams.q_lr,
                                            policy_lr=agent_hyperparams.policy_lr,
                                            num_envs=agent_hyperparams.num_envs,
                                            num_steps=agent_hyperparams.num_steps,
                                            gamma=agent_hyperparams.gamma,
                                            update_epochs=agent_hyperparams.update_epochs))
    collector.set_agent(model)

    # torchexplorer.watch(model, backend="wandb")
    wandb.watch((model, model._actor, model._critic), log="all", log_freq=1000, log_graph=False)

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
                                     model = model)
    callbacks.append(CheckpointCallbackRB(save_path=run_folder+"/checkpoints",
                                          model=model,
                                          save_best=False,
                                          save_freq_ep=checkpoint_freq*agent_hyperparams.num_envs))
    model.save(folderName+"/model_untrained.zip")

    ggLog.info(f"Starting training.")
    try:
        train_on_policy(collector=collector,
            model = model,
            callbacks=callbacks,
            num_steps=agent_hyperparams.num_steps,
            storage_torch_device=agent_hyperparams.th_device,
            train_steps=agent_hyperparams.total_steps)
    finally:
        collector.close()

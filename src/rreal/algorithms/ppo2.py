# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from rreal.algorithms.sac_helpers import gym_builder, build_vec_env, wrap_with_logger, env_builder2vec, build_eval_callbacks
import adarl.utils.session
import inspect
import copy
from rreal.algorithms.sac_helpers import EnvBuilderProtocol, VecEnvBuilderProtocol
from typing import Any, Final
from adarl.utils.callbacks import TrainingCallback, CallbackList, CheckpointCallbackRB
import adarl.utils.sigint_handler
from rreal.algorithms.rl_agent import RLAgent
from typing_extensions import override
from rreal.feature_extractors.feature_extractor import FeatureExtractor
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor
from adarl.utils.utils import numpy_to_torch_dtype_dict
from adarl.utils.tensor_trees import map_tensor_tree
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
from adarl.utils.wandb_wrapper import wandb_log

def layer_init(layer, std=2, bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPORolloutBuffer():
    def __init__(self, num_steps, num_envs, obs_space : gym.spaces.Dict, act_space_shape, device):
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
        self._dones = th.zeros((num_steps+1, num_envs), device=device)
        self._values = th.zeros((num_steps+1, num_envs), device=device)
        self._pos = 0

    
    def add(self, start_obss, actions, logprobs, rewards, prev_dones, values):
        for k in start_obss.keys():
            self._obs[k][self._pos] = start_obss[k]
        self._actions[self._pos] = actions
        self._rewards[self._pos] = rewards
        self._dones[self._pos] = prev_dones
        self._values[self._pos] = values
        self._logprobs[self._pos] = logprobs
        self._pos += 1

    
    def set_logprobs_values(self, logprobs, values):
        self._values[self._pos] = values
        self._logprobs[self._pos] = logprobs


    def set(self, start_obss=None, actions=None, logprobs=None, rewards=None, prev_dones=None, values=None):
        if start_obss is not None:
            for k in start_obss.keys():
                self._obs[k][self._pos] = start_obss[k]
        if actions is not None:
            self._actions[self._pos] = actions
        if rewards is not None:
            self._rewards[self._pos] = rewards
        if prev_dones is not None:
            self._dones[self._pos] = prev_dones
        if values is not None:
            self._values[self._pos] = values
        if logprobs is not None:
            self._logprobs[self._pos] = logprobs

    def reset(self):
        self._pos = 0

    def get_rollout_data(self):
        return (self._obs, #[:self._pos+1],
                self._actions, #[:self._pos],
                self._rewards, #[:self._pos],
                self._dones, #[:self._pos+1],
                self._logprobs, #[:self._pos+1])
                self._values) #[:self._pos+1],
    

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

class PPO(RLAgent):

    @dataclass(repr=False, eq=False)
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

    _hp : Final[Hyperparams]

    def __init__(self, hyperparams : Hyperparams,
                 feature_extractor : FeatureExtractor | None = None):
        super().__init__()
        self._hp = copy.deepcopy(hyperparams)
        if feature_extractor is None:
            self._feature_extractor = StackVectorsFeatureExtractor(observation_space=self._hp.observation_space)
        else:
            raise NotImplementedError()
            self._feature_extractor = feature_extractor
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self._feature_extractor.encoding_size(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self._hp.th_device)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self._feature_extractor.encoding_size(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self._hp.action_len), std=0.01),
        ).to(self._hp.th_device)
        self.actor_logstd = nn.Parameter(th.zeros(1, self._hp.action_len, device=self._hp.th_device))
        self._optimizer = optim.Adam(self.parameters(), lr=self._hp.policy_lr, eps=1e-5)
        self._grad_step_count = th.as_tensor(0, device=self._hp.th_device)
        if self._hp.num_envs*self._hp.num_steps % self._hp.minibatch_size != 0:
            raise RuntimeError(f"num_envs*num_steps must be a multiple of minibatch_size, but num_envs={self._hp.num_envs}, num_steps={self._hp.num_steps} and minibatch_size={self._hp.minibatch_size}")
        self.__batch_size : Final[int] = int(self._hp.num_envs*self._hp.num_steps)
        self.__minibatch_num : Final[int] = int(self.__batch_size/self._hp.minibatch_size)

        self.__stats = { "tot_grad_steps_count":th.as_tensor(0, device=self._hp.th_device),
                        "q_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "actor_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "entropy_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_q_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_actor_loss":th.as_tensor(float("nan"), device=self._hp.th_device),
                        "avg_entropy_loss":th.as_tensor(float("nan"), device=self._hp.th_device)}

    @override
    def input_device(self):
        return self._hp.th_device

    def get_value(self, x):
        return self.critic(x)

    def get_action_logprob_entropy_critic_mean(self, obs_batch, enc_obs_batch = None, action=None):
        if enc_obs_batch is None:
            enc_obs_batch = self._feature_extractor.extract_features(obs_batch)
        action_mean = self.actor_mean(enc_obs_batch)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(enc_obs_batch), action_mean

    def compute_losses(self, mb_advantages, mb_values, mb_returns, mb_logprobs, newvalues, newlogprobs, entropies):
        logratio = newlogprobs - mb_logprobs                
        ratio = logratio.exp()
        if self._hp.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self._hp.clip_coef, 1 + self._hp.clip_coef)
        policy_loss = th.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = newvalues.view(-1)
        if self._hp.clip_vloss:
            v_loss_unclipped = (newvalues - mb_returns) ** 2
            v_clipped = mb_values + th.clamp(
                newvalues - mb_values,
                -self._hp.clip_coef,
                self._hp.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
        else:
            value_loss = 0.5 * ((newvalues - mb_returns) ** 2).mean()

        entropy_loss = entropies.mean()
        loss = policy_loss - self._hp.ent_coef * entropy_loss + value_loss * self._hp.vf_coef
        return loss, policy_loss, value_loss, entropy_loss

    @override
    @th.jit.export
    def train_model(self, buff : PPORolloutBuffer):
        # t_0 = time.monotonic()
        raw_obss, actions, rewards, dones, logprobs, values = buff.get_rollout_data()
        enc_obss = self._feature_extractor.extract_features(raw_obss) 
        
        # t_postenc = time.monotonic()
        # bootstrap value if not done
        with th.no_grad():
            advantages = th.zeros_like(rewards).to(self._hp.th_device)
            lastgaelam = 0
            for t in reversed(range(self._hp.num_steps)):
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
                delta = rewards[t] + self._hp.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self._hp.gamma * self._hp.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values[:-1]
        # t_postadv = time.monotonic()
        
        # flatten the batch
        minibatch_size = self._hp.minibatch_size
        b_encobs = enc_obss[:self._hp.num_steps].view((self.__batch_size,) + enc_obss.size()[2:])
        b_logprobs = logprobs[:self._hp.num_steps].view(self.__batch_size)
        b_actions = actions.view((self.__batch_size,self._hp.action_len))
        b_advantages = advantages.view(self.__batch_size)
        b_returns = returns.view(self.__batch_size)
        b_values = values[:self._hp.num_steps].view(self.__batch_size)

        b_inds = th.zeros(size=(self.__batch_size,), dtype=th.long, device=self._hp.th_device)
        # clipfracs = []
        # t_pretrain = time.monotonic()
        policy_losses = th.empty(size=(self._hp.update_epochs*int(self.__batch_size/self._hp.minibatch_size),), device=self._hp.th_device)
        value_losses = th.empty(size=(self._hp.update_epochs*int(self.__batch_size/self._hp.minibatch_size),), device=self._hp.th_device)
        entropy_losses = th.empty(size=(self._hp.update_epochs*int(self.__batch_size/self._hp.minibatch_size),), device=self._hp.th_device)
        for epoch in range(self._hp.update_epochs):
            th.randperm(self.__batch_size, out=b_inds, device=self._hp.th_device)
            # t_it0 = time.monotonic()
            for i in range(self.__minibatch_num):
                start = i*minibatch_size
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                # print(f"mb_inds = {mb_inds}")
                mb_encobs = b_encobs[mb_inds]
                mb_acts = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_values = b_values[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]

                _, newlogprobs, entropies, newvalues, _ = self.get_action_logprob_entropy_critic_mean(obs_batch=None, enc_obs_batch=mb_encobs, action=mb_acts)

                # with th.no_grad():
                #     old_approx_kl = (-logratio).mean()
                #     approx_kl = ((ratio - 1) - logratio).mean()
                #     clipfracs += [((ratio - 1.0).abs() > self._hp.clip_coef).float().mean().item()]

                loss, policy_loss, value_loss, entropy_loss = self.compute_losses(   mb_advantages = mb_advantages,
                                            mb_values = mb_values,
                                            mb_returns = mb_returns,
                                            mb_logprobs = mb_logprobs, 
                                            newvalues = newvalues,
                                            newlogprobs = newlogprobs,
                                            entropies = entropies)
                policy_losses[epoch*self.__minibatch_num+i] = policy_loss
                value_losses[epoch*self.__minibatch_num+i] = value_loss
                entropy_losses[epoch*self.__minibatch_num+i] = entropy_loss

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self._hp.max_grad_norm)
                self._optimizer.step()
                self._grad_step_count += 1
        
            # if self._hp.target_kl is not None and approx_kl > self._hp.target_kl:
            #     break


        self.__stats.update({"tot_grad_steps_count":self._grad_step_count,
                            "q_loss":value_loss,
                            "actor_loss":policy_loss,
                            "entropy_loss":entropy_loss,
                            "avg_q_loss":th.mean(value_losses),
                            "avg_actor_loss":th.mean(policy_losses),
                            "avg_entropy_loss":th.mean(entropy_losses)})        # t_f = time.monotonic()
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
    def predict_action(self, observation_batch, deterministic = False):
        if deterministic:
            act, logprob, entropy, critic, mean = self.get_action_logprob_entropy_critic_mean(obs_batch=observation_batch)
            return mean
        else:
            act, logprob, entropy, critic, mean = self.get_action_logprob_entropy_critic_mean(obs_batch=observation_batch)
            return act

    @override
    def save(self, path : str):
        pass

    @override
    def load_(self, path : str):
        pass
    
    @override
    def load(cls, path : str):
        pass

        

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

        if not isinstance(action_space, (gym.spaces.Box, gym.spaces.box.Box)):
            raise NotImplementedError(f"unsupported action space {action_space} of type {type(action_space)}")
        self._single_action_space = action_space
        # if not isinstance(obs_space, gym.spaces.Dict):
        #     raise NotImplementedError(f"unsupported observation space {obs_space}")
        self._single_observation_space = obs_space

        self._latest_obs, _ = self._vec_env.reset(seed=seed)
        # self._latest_obs = th.Tensor(self._latest_obs).to(th_device)
        self._latest_done = th.zeros(self._num_envs).to(th_device)
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
            for step in range(0, vsteps_to_collect):
                # ALGO LOGIC: action logic
                start_obs = self._latest_obs
                prev_done = self._latest_done
                th_start_obs = map_tensor_tree(start_obs, lambda a: th.as_tensor(a, device = agent.input_device()))
                action, logprob, _, value, _ = agent.get_action_logprob_entropy_critic_mean(obs_batch=th_start_obs)

                # TRY NOT TO MODIFY: execute the game and log data.
                action = action.to(device=self._env_device)
                next_obs, reward, terminations, truncations, info = self._vec_env.step(action)
                next_obs, reward, terminations, truncations = map_tensor_tree((next_obs, reward, terminations, truncations), lambda a: th.as_tensor(a))
                next_obs = map_tensor_tree(next_obs, lambda t: t.detach().clone())
                next_done = th.logical_or(terminations, truncations)
                term_count += th.count_nonzero(next_done)
                buffer.add(start_obss=start_obs,
                            values=value.flatten(),
                            prev_dones=prev_done,
                            actions=action,
                            logprobs=logprob,
                            rewards=th.as_tensor(reward, device=self._device).view(-1))
                # buffer._obs[step] = start_obs
                self._latest_obs = next_obs
                self._latest_done = next_done

            self._latest_obs = map_tensor_tree(self._latest_obs, lambda a: th.as_tensor(a, device = agent.input_device()))
            action, logprob, _, value, _ = agent.get_action_logprob_entropy_critic_mean(obs_batch=self._latest_obs)
            buffer.set(start_obss=self._latest_obs,
                        values=value.flatten(),
                        prev_dones=self._latest_done,
                        logprobs=logprob)
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
        adarl.utils.session.default_session.run_info["train_iterations"].value = model._grad_step_count.item()
        t_f = time.monotonic()
        t_train_sl += t_f-t_post_cb
        t_tot_sl += t_f - t0

        wlogs = {"ppo/"+k:v for k,v in model.get_stats().items()}
        wandb_log(wlogs,throttle_period=1)
        if global_step - last_log_steps > log_freq_vstep*collector.num_envs():
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
    log_freq_vstep : int

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

    collector = Collector(  vec_env_builder=vec_env_builder,
                            env_builder_args=env_builder_args,
                            th_device=agent_hyperparams.th_device,
                            run_folder=run_folder,
                            num_envs=agent_hyperparams.num_envs,
                            seed=seed)
    action_space = collector.single_action_space()
    obs_space = collector.single_observation_space()
    agent = PPO(PPO.Hyperparams(minibatch_size=agent_hyperparams.minibatch_size,
                                    th_device=agent_hyperparams.th_device,
                                    action_len=int(np.prod(action_space.shape)),
                                    observation_space=obs_space,
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
                                          save_freq_ep=checkpoint_freq*agent_hyperparams.num_envs))

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
                agent_hyperparams=PPO_hyperparams(  minibatch_size=512,
                                                    th_device=th.device("cuda"),
                                                    policy_arch=None,
                                                    q_network_arch=None,
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
                checkpoint_freq=-1,
                collector_device=th.device("cpu"))





if __name__ == "__main__":
    example()
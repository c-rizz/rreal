
from __future__ import annotations
import time
from dataclasses import dataclass, asdict

import gymnasium as gym
import adarl.utils.callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as th
import adarl.utils.session
from adarl.utils.buffers import ThDReplayBuffer, TransitionBatch
from typing import List, Union
from rreal.utils import build_mlp_net
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.sigint_handler
import adarl.utils.session as session
from adarl.utils.wandb_wrapper import wandb_log
from adarl.utils.callbacks import TrainingCallback, CallbackList
from rreal.algorithms.collectors import ExperienceCollector
from rreal.algorithms.rl_policy import RLPolicy
import inspect
import yaml
from adarl.utils.tensor_trees import sizetree_from_space, map2_tensor_tree, flatten_tensor_tree, map_tensor_tree
from rreal.feature_extractors.stack_vectors_feature_extractor import StackVectorsFeatureExtractor
from rreal.feature_extractors.feature_extractor import FeatureExtractor
class QNetwork(nn.Module):
    def __init__(self,
                 action_size : int,
                 q_network_arch : List[int],
                 feature_extractor : FeatureExtractor,
                 torch_device : Union[str,th.device] = "cuda",
                 nets_num : int = 1):
        super().__init__()
        self._nets_num = nets_num
        self._feature_extractor = feature_extractor
        self._q_nets = build_mlp_net(arch=q_network_arch,
                                     input_size=action_size + self._feature_extractor.encoding_size(),
                                     output_size=1,
                                     ensemble_size=self._nets_num,
                                     return_ensemble_mean=False).to(device=torch_device)
    
    def get_min_qval(self, observations, actions):
        qvals = self(observations, actions)
        # ggLog.info(f"qvals.size() = {qvals.size()}")
        min_q = th.min(qvals,dim=1).values
        # min_q = min_q.squeeze(1)
        # ggLog.info(f"min_q.size() = {min_q.size()}")
        return min_q
    
    def forward(self, observations, actions):
        observations = self._feature_extractor.extract_features(observation_batch=observations)
        qvals = self._q_nets(torch.cat([observations, actions], 1))
        return qvals



class Actor(nn.Module):
    def __init__(self,  action_size,
                        feature_extractor : FeatureExtractor,
                        policy_arch = [256,256],
                        action_max : Union[float, th.Tensor] = 1,
                        action_min : Union[float, th.Tensor] = -1,
                        log_std_max = 2,
                        log_std_min = -5,
                        torch_device : Union[str,th.device] = "cuda"):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self.device = torch_device
        self._feature_extractor = feature_extractor
        if len(policy_arch)<1:
            raise RuntimeError(f"Invalid policy arch {policy_arch}, must have at least 1 layer")
        else:
            self.act_fc = build_mlp_net(arch=policy_arch[:-1],input_size=self._feature_extractor.encoding_size(), output_size=policy_arch[-1],
                                    last_activation_class=th.nn.LeakyReLU).to(device=torch_device)
        self.act_fc_mean = nn.Linear(policy_arch[-1], action_size, device=torch_device)
        self.act_fc_logstd = nn.Linear(policy_arch[-1], action_size, device=torch_device)

        if isinstance(action_max, int): action_max = float(action_max)
        if isinstance(action_min, int): action_min = float(action_min)
        if isinstance(action_max,float): action_max = th.as_tensor([action_max]*action_size, dtype=th.float32)
        if isinstance(action_min,float): action_min = th.as_tensor([action_min]*action_size, dtype=th.float32)
        # save action scaling factors as non-trained parameters
        self.register_buffer("action_scale", torch.as_tensor((action_max - action_min) / 2.0, dtype=torch.float32, device=torch_device))
        self.register_buffer("action_bias",  torch.as_tensor((action_max + action_min) / 2.0, dtype=torch.float32, device=torch_device))

    def forward(self, observation_batch):
        observation_batch = self._feature_extractor.extract_features(observation_batch=observation_batch)
        observation_batch = self.act_fc(observation_batch)
        mean = self.act_fc_mean(observation_batch)
        log_std = self.act_fc_logstd(observation_batch)
        log_std = (torch.tanh(log_std)+1)*0.5*(self._log_std_max - self._log_std_min) + self._log_std_min
        return mean, log_std

    def sample_action(self, observation_batch):
        mean, log_std = self(observation_batch)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # squash the action in [-1,1]
        log_prob = normal.log_prob(x_t) # get the probability of the actions that we sampled

        # scale mean and action to the proper bounds
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # correct the probability for the squashing
        log_prob = log_prob.sum(1, keepdim=True) # get probability per each multidimensional action, not for each action component

        return action, log_prob, mean


class SAC(RLPolicy):
    @dataclass
    class Hyperparams():
        q_lr : float
        policy_lr : float
        gamma : float
        auto_entropy_temperature : bool
        constant_entropy_temperature : float | None
        action_size : int
        action_min : th.Tensor
        action_max : th.Tensor
        target_tau : float
        policy_update_freq : int
        targets_update_freq : int
        q_network_arch : List[int]
        policy_arch : List[int]
        torch_device : Union[str,th.device]
        target_entropy : float | None
        observation_space : gym.spaces.Space

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
                 target_entropy : float | None = None,
                 gamma : float = 0.99,
                 target_tau = 0.005,
                 policy_update_freq = 2,
                 target_update_freq = 1,
                 feature_extractor : FeatureExtractor | None = None):
        super().__init__()
        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        self._init_args = values
        self._init_args.pop("self")
        self._init_args.pop("__class__")
        self._hp = SAC.Hyperparams(q_lr=q_lr,
                                   policy_lr = policy_lr,
                                   gamma=gamma,
                                   auto_entropy_temperature=auto_entropy_temperature,
                                   constant_entropy_temperature=constant_entropy_temperature,
                                   action_size=action_size,
                                   action_min = th.as_tensor(action_min),
                                   action_max = th.as_tensor(action_max),
                                   target_tau = target_tau,
                                   policy_update_freq=policy_update_freq,
                                   targets_update_freq=target_update_freq,
                                   q_network_arch = q_network_arch,
                                   policy_arch = policy_arch,
                                   torch_device = torch_device,
                                   target_entropy = target_entropy,
                                   observation_space = observation_space)
        self._obs_space_sizes = sizetree_from_space(observation_space)
        self.device = torch_device
        self._value_func_updates = 0
        self._policy_updates = 0
        if feature_extractor is None:
            self._feature_extractor = StackVectorsFeatureExtractor(observation_space=observation_space)
        else:
            self._feature_extractor = feature_extractor
        self._q_net = QNetwork( action_size=self._hp.action_size,
                                feature_extractor=self._feature_extractor,
                                q_network_arch=q_network_arch,
                                torch_device=self._hp.torch_device,
                                nets_num=2)
        self._q_net_target = QNetwork(  feature_extractor=self._feature_extractor,
                                        action_size=self._hp.action_size,
                                        q_network_arch=q_network_arch,
                                        torch_device=self._hp.torch_device,
                                        nets_num=2)
        self._q_net_target.load_state_dict(self._q_net.state_dict())
        self._q_optimizer = optim.Adam(self._q_net.parameters(), lr=self._hp.q_lr)
        self._actor = Actor(feature_extractor=self._feature_extractor,
                            policy_arch=policy_arch,
                            action_size = self._hp.action_size,
                            action_min = self._hp.action_min,
                            action_max = self._hp.action_max,
                            torch_device=self._hp.torch_device)
        self._actor_optimizer = optim.Adam(list(self._actor.parameters()), lr=self._hp.policy_lr)
        if self._hp.auto_entropy_temperature:
            if self._hp.target_entropy is None:
                self._target_entropy = -self._hp.action_size
            else:
                self._target_entropy = self._hp.target_entropy
            self._log_alpha = torch.zeros(1, requires_grad=True, device=torch_device)
            self._alpha = self._log_alpha.exp().item()
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hp.q_lr)
        else:
            self._alpha = constant_entropy_temperature

        self._last_q_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_actor_loss = th.as_tensor(float("nan"), device=self.device)
        self._last_alpha_loss = th.as_tensor(float("nan"), device=self.device)
        self._tot_grad_steps_count = 0

    def save(self, path : str):
        th.save(self.state_dict(), path)
        extra = {}
        extra["init_args"] = self._init_args
        extra["hyperparams"] = asdict(self._hp)
        with open(path+".extra.yaml", "w") as init_args_yamlfile:
            yaml.dump(extra,init_args_yamlfile, default_flow_style=None)

    @staticmethod
    def load(path : str):
        with open(path+".extra.yaml", "r") as init_args_yamlfile:
            extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        model = SAC(**extra["init_args"])
        model._hp = SAC.Hyperparams(**extra["hyperparams"]) # shouldn't be necessary, but shouldn't hurt
        model.load_state_dict(th.load(path))
        return model

    def predict_action(self, observation_batch, deterministic = False):
        # s = {k:v.size() for k,v in observation.items()}
        # ggLog.info(f"predict: observation = {s}")


        # check if it is not a batch, if so, unsqueeze
        d = map2_tensor_tree(observation_batch, self._obs_space_sizes,
                        lambda obs,obs_space_size: obs.dim() == len(obs_space_size))
        batched = not all(flatten_tensor_tree(d).values())
        if not batched:
            observation_batch = map_tensor_tree(observation_batch, lambda t: t.unsqueeze(0))
        observation_batch = map_tensor_tree(observation_batch, lambda t: t.to(device = self.device, dtype = th.float32))

        action, log_prob, mean = self._actor.sample_action(observation_batch)
        if not batched:
            action = action.squeeze()
            mean = mean.squeeze()
            log_prob = log_prob.squeeze()
        if deterministic:
            return mean
        else:
            return action
        
    def get_hidden_state(self):
        return None

    def _update_value_func(self, transitions : TransitionBatch):
        with torch.no_grad():
            # Compute next-values for TD
            next_state_actions, next_state_log_pi, _ = self._actor.sample_action(transitions.next_observations)
            q_next = self._q_net_target.get_min_qval(transitions.next_observations, next_state_actions)
            # ggLog.info(f"next_state_log_pi.size() = {next_state_log_pi.size()}")
            soft_q_next = q_next - self._alpha * next_state_log_pi
            # ggLog.info(f"soft_q_next.size() = {soft_q_next.size()}")
            td_q_values = transitions.rewards.flatten() + (1 - transitions.terminated.flatten()) * self._hp.gamma * (soft_q_next).view(-1)

        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        q_values = self._q_net(transitions.observations, transitions.actions)
        # ggLog.info(f"q_values.size() = {q_values.size()}")
        td_q_values = td_q_values.unsqueeze(1).unsqueeze(2)
        # assert td_q_values.size() == (q_values.size()[0], 1, 1)
        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        td_q_values = td_q_values.expand(-1,2,1)
        # ggLog.info(f"td_q_values.size() = {td_q_values.size()}")
        q_loss = F.mse_loss(q_values, td_q_values)

        self._q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self._q_optimizer.step()
        self._value_func_updates += 1
        self._last_q_loss = q_loss.detach()

    def _update_policy(self, transitions : TransitionBatch):
        act, act_log_prob, _ = self._actor.sample_action(transitions.observations)
        min_q_pi = self._q_net.get_min_qval(transitions.observations, act) # cannot reuse those from _update_value_func, the value function has changed
        # ggLog.info(f"min_q_pi.size() = {min_q_pi.size()}")
        # ggLog.info(f"act_log_prob.size() = {act_log_prob.size()}")
        actor_loss = ((self._alpha * act_log_prob) - min_q_pi).mean()

        self._actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._actor_optimizer.step()

        if self._hp.auto_entropy_temperature:
            with torch.no_grad():
                _, act_log_prob, _ = self._actor.sample_action(transitions.observations)
            alpha_loss = (-self._log_alpha.exp() * (act_log_prob + self._target_entropy)).mean()

            self._alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._alpha = self._log_alpha.exp().item()
        self._policy_updates += 1
        self._last_actor_loss = actor_loss.detach()
        self._last_alpha_loss = alpha_loss.detach()

    @staticmethod
    def _target_update(param, target_param, tau):
            if tau == 1:
                target_param.data.copy_(param.data)
            else:
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
    def _update_target_nets(self):
        for param, target_param in zip(self._q_net.parameters(), self._q_net_target.parameters()):
            self._target_update(param, target_param, self._hp.target_tau)

    def update(self, transitions : TransitionBatch):
        self._update_value_func(transitions = transitions)
        if self._value_func_updates % self._hp.policy_update_freq == 0:
            for _ in range(self._hp.policy_update_freq):
                self._update_policy(transitions=transitions)
        if self._value_func_updates % self._hp.targets_update_freq == 0:
            self._update_target_nets()
        return self._last_q_loss, self._last_actor_loss, self._last_alpha_loss

    def train(self, global_step, learning_starts, iterations, batch_size, buffer):
        q_loss, actor_loss, alpha_loss = float("nan"), float("nan"), float("nan")
        if global_step > learning_starts:
            q_act_alpha_losses = th.zeros(size=(iterations, 3), dtype=th.float32, device=self.device)
            for i in range(iterations):
                data = buffer.sample(batch_size)
                nq_loss, nactor_loss, nalpha_loss = self.update(transitions = data)
                q_act_alpha_losses[i] = th.stack((nq_loss,nactor_loss,nalpha_loss))
                self._tot_grad_steps_count += 1
            # q_loss, actor_loss, alpha_loss = q_act_alpha_losses.mean(dim = 0).cpu().numpy()
            q_loss, actor_loss, alpha_loss = q_act_alpha_losses[-1].cpu().numpy()
            wandb_log({"sac/tot_grad_steps_count":self._tot_grad_steps_count,
                        "sac/q_loss":q_loss,
                        "sac/actor_loss":actor_loss,
                        "sac/alpha_loss":alpha_loss,
                        "sac/alpha":self._alpha},
                        throttle_period=2)
        print(f"SAC: gsteps={self._tot_grad_steps_count} q_loss={q_loss:5g} actor_loss={actor_loss:5g} alpha_loss={alpha_loss:5g}")


def train_off_policy(collector : ExperienceCollector,
          model : SAC,
          buffer : ThDReplayBuffer,
          total_timesteps : int,
          train_freq : int,
          learning_starts : int,
          grad_steps : int,
          batch_size : int,
          log_freq : int = -1,
          callbacks : Union[TrainingCallback, List[TrainingCallback]] | None = None):
    if log_freq == -1: log_freq = train_freq
    num_envs = collector.num_envs()

    collector.reset()
    global_step = 0
    t_coll_sl = 0
    t_train_sl = 0
    t_tot_sl = 0
    
    if callbacks is None:
        callbacks = []
    if not isinstance(callbacks, CallbackList):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]    
        callbacks = CallbackList(callbacks=callbacks)

    callbacks.on_training_start()
    ep_counter = 0
    step_counter = 0
    while global_step < total_timesteps and not session.default_session.is_shutting_down():
        s0b = buffer.stored_frames()
        t0 = time.monotonic()

        # ------------------  Start experience collection  ------------------
        steps_to_collect = train_freq*num_envs
        vsteps_to_collect = train_freq
        callbacks.on_collection_start()
        collector.collect_experience_async(model_state_dict=model.state_dict(),
                                            vsteps_to_collect=vsteps_to_collect,
                                            global_vstep_count=global_step//num_envs,
                                            random_vsteps=learning_starts//num_envs)

        # ------------------             Train             ------------------
        t_before_train = time.monotonic()
        model.train(global_step, learning_starts, grad_steps, batch_size, buffer)
        t_after_train = time.monotonic()

        # ------------------   Store collected experience  ------------------
        tmp_buff = collector.wait_collection(timeout = 120.0)
        new_episodes = tmp_buff.added_completed_episodes() - ep_counter
        ep_counter = tmp_buff.added_completed_episodes()
        step_counter = tmp_buff.added_frames()
        t_coll_sl += collector.collection_duration()
        adarl.utils.session.default_session.run_info["collected_episodes"] = ep_counter
        adarl.utils.session.default_session.run_info["collected_steps"] = step_counter
        # callbacks._callbacks[0].set_model(model)
        callbacks.on_collection_end(collected_steps=vsteps_to_collect*num_envs,
                                   collected_episodes=new_episodes,
                                   collected_data=tmp_buff)
        s = 0
        for (obs, next_obs, action, reward, terminated, truncated) in tmp_buff.replay():
            # ggLog.info(f"replaying step {s} ")
            s+=1
            buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward,
                        truncated=truncated, terminated=terminated)

        # ------------------      Wrap up and restart      ------------------
        if buffer.stored_frames()-s0b != steps_to_collect and not buffer.full:
            raise RuntimeError(f"Expected to collect {steps_to_collect} but got {buffer.stored_frames()-s0b}")
        global_step += steps_to_collect
        tf = time.monotonic()
        t_train_sl += t_after_train - t_before_train
        t_tot_sl += tf-t0
        if global_step/num_envs % log_freq == 0:
            ggLog.info(f"OFFTRAIN: expstps:{global_step} trainstps={model._tot_grad_steps_count} coll={t_coll_sl:.2f}s train={t_train_sl:.2f}s tot={t_tot_sl:.2f}")
            t_train_sl, t_coll_sl, t_tot_sl = 0,0,0
            adarl.utils.sigint_handler.haltOnSigintReceived()
    callbacks.on_training_end()

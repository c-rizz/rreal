#!/usr/bin/env python3  

from __future__ import annotations
import os
import random
import time

import numpy as np
import torch
import torch as th
# import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
from adarl.utils.async_vector_env import AsyncVectorEnvShmem
import inspect
import adarl.utils.session
from adarl.envs.vector_env_logger import VectorEnvLogger
from adarl.utils.buffers import ThDReplayBuffer
import adarl.utils.sigint_handler
from rreal.algorithms.sac import SAC, train_off_policy
from rreal.algorithms.collectors import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector
import wandb 
from adarl.utils.callbacks import EvalCallback, CheckpointCallbackRB
import gymnasium as gym
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.GymToLr import GymToLr
from adarl.envs.lr_wrappers.ObsToDict import ObsToDict
from rreal.tmp.gym_transform_observation import DtypeObservation
import copy

def gym_builder(seed, log_folder, env_builder_args):
    os.environ["MUJOCO_GL"]="egl"
    env_kwargs : dict = copy.deepcopy(env_builder_args)
    env_kwargs.pop("env_name")
    gymenv = gym.make(env_builder_args["env_name"],
                    render_mode="rgb_array",
                    **env_kwargs)
    gymenv = DtypeObservation(gymenv, dtype=np.float32)
    lrenv = GymToLr(gymenv,
                    stepSimDuration_sec=0.05,
                    maxStepsPerEpisode=env_builder_args["max_episode_steps"],
                    copy_observations=True,
                    actions_to_numpy=True)
    lrenv = ObsToDict(env=lrenv)
    lrenv.seed(seed=seed)
    
    return GymEnvWrapper(env=lrenv,
                        episodeInfoLogFile = log_folder+f"/GymEnvWrapper_log.{seed:010d}.csv")

    

def build_vec_env(env_builder, env_builder_args, log_folder, seed, num_envs) -> gym.vector.VectorEnv:
    builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(num_envs)]
    envs = AsyncVectorEnvShmem(builders, context="forkserver", purely_numpy=False, shared_mem_device = th.device("cpu"), copy_data=False)
    envs = VectorEnvLogger(env = envs)
    return envs

def build_sac(obs_space : gym.Space, act_space : gym.Space, hyperparams):
    return SAC(observation_space=obs_space,
                action_size=int(np.prod(act_space.shape)),
                q_network_arch=hyperparams.q_network_arch,
                q_lr=hyperparams.q_lr,
                policy_lr=hyperparams.policy_lr,
                policy_arch=hyperparams.policy_network_arch,
                action_min = act_space.low.tolist(),
                action_max = act_space.high.tolist(),
                torch_device=hyperparams.device,
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=hyperparams.gamma,
                target_tau = hyperparams.target_tau,
                policy_update_freq=2,
                target_update_freq=1,
                batch_size = hyperparams.batch_size)


from dataclasses import dataclass
@dataclass
class SAC_hyperparams:
    q_network_arch : list[int]
    policy_network_arch : list[int]
    q_lr : float
    policy_lr : float
    device : th.device
    gamma : float
    target_tau : float
    buffer_size : int
    total_steps : int
    train_freq : int
    learning_starts : int
    grad_steps : int
    batch_size : int
    parallel_envs : int
    log_freq_vstep : int
    eval_freq_ep : int


def solve_sac(seed, folderName, run_id, args, env_builder, env_builder_args, hyperparams : SAC_hyperparams):

    log_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams.device = device
    # env setup
    num_envs = hyperparams.parallel_envs
    use_processes = True
    if use_processes:
        collector = AsyncProcessExperienceCollector(
                            vec_env_builder=lambda: build_vec_env(env_builder, env_builder_args=env_builder_args,
                                                                log_folder=log_folder,
                                                                seed=seed,
                                                                num_envs=num_envs),
                            storage_torch_device=device,
                            buffer_size=hyperparams.train_freq*num_envs,
                            session=session)
    else:
        collector = AsyncThreadExperienceCollector( vec_env=build_vec_env(env_builder, env_builder_args=env_builder_args,
                                                                log_folder=log_folder,
                                                                seed=seed,
                                                                num_envs=num_envs),
                                                    buffer_size=hyperparams.train_freq*num_envs,
                                                    storage_torch_device=device)
    observation_space = collector.observation_space()
    action_space = collector.action_space()
    collector.set_base_collector_model(lambda o,a: build_sac(o,a,hyperparams))
    model = build_sac(observation_space, action_space, hyperparams)

    # torchexplorer.watch(model, backend="wandb")
    wandb.watch((model, model._actor, model._q_net), log="all", log_freq=1000, log_graph=True)

    # compiled_model = th.compile(model)

    rb = ThDReplayBuffer(
        buffer_size=hyperparams.buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        storage_torch_device=device,
        handle_timeout_termination=True,
        n_envs=num_envs)
    
    start_time = time.time()

    eval_env = gym_builder(log_folder=log_folder+"/eval",
                                    seed=seed+100000000,
                                    env_builder_args = env_builder_args)
    fps = 1/eval_env.getBaseEnv().env._stepSimDuration_sec #ugly, sorry
    eval_env_rec = RecorderGymWrapper(  env=eval_env,
                                fps = fps,
                                outFolder=log_folder+"/eval/videos/RecorderGymWrapper")
    eval_env_rec_det = RecorderGymWrapper(  env=eval_env,
                                fps = fps,
                                outFolder=log_folder+"/eval_deterministic/videos/RecorderGymWrapper")
    callbacks = []
    callbacks.append(EvalCallback(eval_env=eval_env_rec,
                                  model=model,
                                  n_eval_episodes=1,
                                  eval_freq_ep=hyperparams.eval_freq_ep*num_envs,
                                  deterministic=False))
    callbacks.append(EvalCallback(eval_env=eval_env_rec_det,
                                  model=model,
                                  n_eval_episodes=1,
                                  eval_freq_ep=hyperparams.eval_freq_ep*num_envs,
                                  deterministic=True))
    callbacks.append(CheckpointCallbackRB(save_path=log_folder+"/checkpoints",
                                          model=model,
                                          save_best=False,
                                          save_freq_ep=100*num_envs))
    try:
        train_off_policy(collector=collector,
            model = model,
            buffer = rb,
            total_timesteps=hyperparams.total_steps,
            train_freq = hyperparams.train_freq,
            learning_starts=hyperparams.learning_starts,
            grad_steps=hyperparams.grad_steps,
            log_freq_vstep=hyperparams.log_freq_vstep,
            callbacks=callbacks)
    finally:
        collector.close()

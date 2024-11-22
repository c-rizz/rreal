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
from adarl.utils.ThDictEpReplayBuffer import ThDictEpReplayBuffer
import adarl.utils.sigint_handler
from rreal.algorithms.sac import SAC, train_off_policy
from rreal.algorithms.collectors import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector
import wandb 
from adarl.utils.callbacks import EvalCallback, CheckpointCallbackRB
import gymnasium as gym
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.GymToLr import GymToLr
from adarl.envs.lr_wrappers.ObsToDict import ObsToDict
from rreal.tmp.gym_transform_observation import DtypeObservation
import adarl.utils.dbg.ggLog as ggLog
import copy
import typing
import adarl.utils.session as session
from rreal.algorithms.rl_agent import RLAgent
from adarl.envs.vector_env_checker import VectorEnvChecker


class EnvBuilderProtocol(typing.Protocol):
    def __call__(self, seed : int, log_folder : str, is_eval : bool, env_builder_args : dict) -> tuple[gym.Env,float]:
        ...

class VecEnvBuilderProtocol(typing.Protocol):
    def __call__(self, seed : int, run_folder : str, num_envs : int, env_builder_args : dict) -> gym.vector.VectorEnv:
        ...

def gym_builder(seed, log_folder, is_eval, env_builder_args : dict[str,typing.Any]):
    os.environ["MUJOCO_GL"]="egl"
    quiet = env_builder_args.pop("quiet",False)
    use_wandb = env_builder_args.pop("use_wandb",True)
    env_kwargs : dict = copy.deepcopy(env_builder_args)
    stepLength_sec = 0.05
    env_kwargs.pop("env_name")
    gymenv = gym.make(env_builder_args["env_name"],
                    render_mode="rgb_array",
                    **env_kwargs)
    gymenv = DtypeObservation(gymenv, dtype=np.float32)
    lrenv = GymToLr(gymenv,
                    stepSimDuration_sec=stepLength_sec,
                    maxStepsPerEpisode=env_builder_args["max_episode_steps"],
                    copy_observations=True,
                    actions_to_numpy=True)
    lrenv = ObsToDict(env=lrenv)
    lrenv.seed(seed=seed)
    
    return GymEnvWrapper(env=lrenv,
                        episodeInfoLogFile = log_folder+f"/GymEnvWrapper_log.{seed:010d}.csv",
                        quiet=quiet,
                        use_wandb = use_wandb), 1/stepLength_sec


def build_vec_env(env_builder_args,
                  log_folder,
                  seed,
                  num_envs,
                  env_builder : EnvBuilderProtocol,
                  purely_numpy : bool = False,
                  logs_id = None,
                  collector_device : th.device = th.device("cuda")) -> gym.vector.VectorEnv:
    if "use_wandb" not in env_builder_args:
        env_builder_args["use_wandb"] = False
    if logs_id is None:
        logs_id = session.default_session.run_info["run_id"]
    builders = [(lambda i: (lambda: env_builder(seed=seed+100000*i,
                                                log_folder=log_folder,
                                                is_eval = False,
                                                env_builder_args = env_builder_args)[0]
                                ))(i) for i in range(num_envs)]
    env = AsyncVectorEnvShmem(builders,
                               context="forkserver",
                               purely_numpy=purely_numpy,
                               shared_mem_device = collector_device,
                               copy_data=False,
                               worker_init_fn=session.set_current_session,
                               worker_init_kwargs={"session":session.default_session})
    env = VectorEnvLogger(env = env,
                           logs_id = logs_id)
    env = VectorEnvChecker(env = env)
    return env

def build_eval_callbacks(eval_configurations : list[dict],
                         vec_env_builder : VecEnvBuilderProtocol,
                         run_folder : str,
                         base_seed : int,
                         collector_device : th.device,
                         model : RLAgent):
    callbacks = []
    for eval_conf in eval_configurations:
        eval_env = vec_env_builder(env_builder_args=eval_conf["env_builder_args"],
                                    run_folder=run_folder+f"/eval_"+eval_conf["name"],
                                    seed=base_seed+100000000,
                                    num_envs=eval_conf["num_envs"])
        callbacks.append(EvalCallback(eval_env=eval_env,
                                    model=model,
                                    n_eval_episodes=eval_conf["eval_eps"],
                                    eval_freq_ep=eval_conf["eval_freq_ep"],
                                    deterministic=eval_conf["deterministic"],
                                    eval_name=eval_conf["name"]))
    return callbacks

# def build_vec_env(env_builder, env_builder_args, log_folder, seed, num_envs) -> gym.vector.VectorEnv:
#     builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
#                                                   seed=seed+100000*i,
#                                                   env_builder_args = env_builder_args)
#                                 ))(i) for i in range(num_envs)]
#     envs = AsyncVectorEnvShmem(builders, context="forkserver", purely_numpy=False, shared_mem_device = th.device("cpu"), copy_data=False)
#     envs = VectorEnvLogger(env = envs)
#     return envs

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
                batch_size = hyperparams.batch_size,
                reference_init_args = hyperparams.reference_init_args)

def build_collector(use_processes : bool,
                    vec_env_builder : VecEnvBuilderProtocol,
                    env_builder_args : dict[str,typing.Any],
                    run_folder : str,
                    seed : int,
                    collector_device : th.device,
                    collector_buffer_size : int,
                    session : adarl.utils.session.Session,
                    num_envs : int):
    vec_env_builder_norags = lambda: vec_env_builder(env_builder_args=env_builder_args,
                                                    run_folder=run_folder,
                                                    seed=seed,
                                                    num_envs=num_envs)
    if use_processes:
        collector = AsyncProcessExperienceCollector(
                            vec_env_builder=vec_env_builder_norags,
                            storage_torch_device=collector_device,
                            buffer_size=collector_buffer_size,
                            session=session)
    else:
        collector = AsyncThreadExperienceCollector( vec_env=vec_env_builder_norags(),
                                                    buffer_size=collector_buffer_size,
                                                    storage_torch_device=collector_device)
    return collector

from dataclasses import dataclass
@dataclass
class SAC_hyperparams:
    q_network_arch : list[int]
    policy_network_arch : list[int]
    q_lr : float
    policy_lr : float
    device : str | th.device
    gamma : float
    target_tau : float
    buffer_size : int
    total_steps : int
    train_freq_vstep : int
    learning_starts : int
    grad_steps : int
    batch_size : int
    parallel_envs : int
    log_freq_vstep : int
    reference_init_args : dict

def vec_env_builder_from_env_builder(env_builder : EnvBuilderProtocol, collector_device : th.device):
    return lambda seed, run_folder, num_envs, env_builder_args: build_vec_env(  env_builder=env_builder,
                                                                                env_builder_args=env_builder_args,
                                                                                log_folder=run_folder,
                                                                                seed=seed,
                                                                                num_envs=num_envs,
                                                                                collector_device=collector_device)

def sac_train(  seed : int,
                folderName : str,
                run_id : str,
                args,
                env_builder : EnvBuilderProtocol | None,
                vec_env_builder : VecEnvBuilderProtocol | None,
                env_builder_args : dict,
                hyperparams : SAC_hyperparams,
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
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if hyperparams.device == "cuda": hyperparams.device = "cuda:0"
    device = th.device(hyperparams.device)
    if collector_device is None:
        collector_device = device
    if vec_env_builder is None and env_builder is not None:
        vec_env_builder = lambda seed, run_folder, num_envs, env_builder_args: build_vec_env(   env_builder=env_builder,
                                                                                                env_builder_args=env_builder_args,
                                                                                                log_folder=run_folder,
                                                                                                seed=seed,
                                                                                                num_envs=num_envs,
                                                                                                collector_device=collector_device)
    if vec_env_builder is None:
        raise RuntimeError(f"You must specify either vec_env_builder or env_builder")
    # env setup
    collector = build_collector(use_processes = True,
                                vec_env_builder = vec_env_builder,
                                env_builder_args = env_builder_args,
                                run_folder = run_folder,
                                seed = seed,
                                num_envs=hyperparams.parallel_envs,
                                collector_device = collector_device,
                                collector_buffer_size = hyperparams.train_freq_vstep*hyperparams.parallel_envs,
                                session = session)
    collector.set_base_collector_model(lambda o,a: build_sac(o,a,hyperparams))    
    observation_space = collector.observation_space()
    action_space = collector.action_space()
    model = build_sac(observation_space, action_space, hyperparams)

    # torchexplorer.watch(model, backend="wandb")
    wandb.watch((model, model._actor, model._q_net), log="all", log_freq=1000, log_graph=False)

    # compiled_model = th.compile(model)

    # rb = ThDReplayBuffer(
    #     buffer_size=hyperparams.buffer_size,
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     device=device,
    #     storage_torch_device=device,
    #     handle_timeout_termination=True,
    #     n_envs=num_envs)
    rb = ThDictEpReplayBuffer(  buffer_size=hyperparams.buffer_size,
                                observation_space=observation_space,
                                action_space=action_space,
                                device=device,
                                storage_torch_device=device,
                                n_envs=hyperparams.parallel_envs,
                                max_episode_duration=max_episode_duration,
                                validation_buffer_size = validation_buffer_size,
                                validation_holdout_ratio = validation_holdout_ratio,
                                min_episode_duration = 0,
                                disable_validation_set = False,
                                fill_val_buffer_to_min_at_step = hyperparams.learning_starts,
                                val_buffer_min_size = validation_batch_size)
    
    ggLog.info(f"Replay buffer occupies {rb.memory_size()/1024/1024:.2f}MB")
    
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
                                          save_freq_ep=checkpoint_freq*hyperparams.parallel_envs))
    model.save(folderName+"/model_untrained.zip")
    try:
        train_off_policy(collector=collector,
            model = model,
            buffer = rb,
            total_timesteps=hyperparams.total_steps,
            train_freq = hyperparams.train_freq_vstep,
            learning_start_step=hyperparams.learning_starts,
            grad_steps=hyperparams.grad_steps,
            log_freq_vstep=hyperparams.log_freq_vstep,
            callbacks=callbacks,
            validation_freq= 1 if validation_enabled else 0,
            validation_batch_size=validation_batch_size)
    finally:
        collector.close()

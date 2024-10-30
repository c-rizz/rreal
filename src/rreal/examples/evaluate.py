#!/usr/bin/env python3  

from __future__ import annotations
import os
import random
import time

import numpy as np
import torch
import torch as th
# import jumping_leg.experiments.build_jumping_leg_env as build_jumping_leg_env
import inspect
import adarl.utils.session
from adarl.utils.buffers import ThDReplayBuffer
import adarl.utils.sigint_handler
from rreal.algorithms.sac import train_off_policy
from rreal.algorithms.collectors import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector
import wandb 
from adarl.utils.callbacks import EvalCallback, CheckpointCallbackRB
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import typing
from rreal.algorithms.rl_agent import RLAgent
from rreal.algorithms.sac_helpers import build_vec_env, EnvBuilderProtocol
from dataclasses import dataclass
from adarl.utils.utils import evaluatePolicyVec
import gymnasium as gym

class ModelBuilderProtocol(typing.Protocol):
    def __call__(self, obs_space : gym.Space, act_space : gym.Space, hyperparams) -> RLAgent:
        ...

def evaluate(seed : int,
              folderName : str,
              run_id : str,
              args,
              env_builder : EnvBuilderProtocol,
              env_builder_args : dict,
              model_builder : ModelBuilderProtocol,
              model_kwargs : dict[str,typing.Any],
              video_recorder_kwargs : dict[str,typing.Any] = {},
              num_envs = 1,
              episodes = 10,
              extra_info_stats = [],
              deterministic = False):

    log_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        use_wandb=False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env setup
    env = build_vec_env(env_builder=env_builder,
                        env_builder_args=env_builder_args,
                        log_folder=log_folder,
                        seed=seed,
                        num_envs=num_envs)
    model = model_builder(env.single_observation_space, env.single_action_space, hyperparams = model_kwargs)
    
    results = evaluatePolicyVec(vec_env=env,
                      model=model,
                      episodes=episodes,
                      extra_info_stats=extra_info_stats,
                      deterministic=deterministic)
    env.close()
    return results
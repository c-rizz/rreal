#!/usr/bin/env python3  

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

def half_cheetah_builder(   log_folder,
                            seed,
                            env_builder_args):
    os.environ["MUJOCO_GL"]="egl"
    gymenv = gym.make('HalfCheetah-v4',
                    forward_reward_weight=env_builder_args["forward_reward_weight"],
                    ctrl_cost_weight=env_builder_args["ctrl_cost_weight"],
                    reset_noise_scale=env_builder_args["reset_noise_scale"],
                    exclude_current_positions_from_observation=env_builder_args["exclude_current_positions_from_observation"],
                    max_episode_steps=env_builder_args["max_episode_steps"],
                    render_mode="rgb_array")
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
    

def build_vec_env(env_builder_args, log_folder, seed, num_envs) -> gym.vector.VectorEnv:
    builders = [(lambda i: (lambda: half_cheetah_builder(log_folder=log_folder,
                                                  seed=seed+100000*i,
                                                  env_builder_args = env_builder_args)
                                ))(i) for i in range(num_envs)]
    envs = AsyncVectorEnvShmem(builders, context="forkserver", purely_numpy=False, shared_mem_device = th.device("cpu"), copy_data=False)
    envs = VectorEnvLogger(env = envs)
    return envs

def build_sac(obs_space : gym.Space, act_space : gym.Space, hyperparams):
    return SAC(observation_space=obs_space,
                action_size=int(np.prod(act_space.shape)),
                q_network_arch=[512,256],
                q_lr=hyperparams["q_lr"],
                policy_lr=hyperparams["policy_lr"],
                policy_arch=[512,256],
                action_min = act_space.low.tolist(),
                action_max = act_space.high.tolist(),
                torch_device=hyperparams["device"],
                auto_entropy_temperature=True,
                constant_entropy_temperature=None,
                gamma=0.99,
                target_tau = 0.005,
                policy_update_freq=2,
                target_update_freq=1)



def main(seed, folderName, run_id, args, env_builder_args, hyperparams):

    log_folder, session = adarl.utils.session.adarl_startup(   __file__,
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        experiment_name=os.path.basename(__file__),
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparams["device"] = device
    # env setup
    num_envs = 16
    use_processes = True
    if use_processes:
        collector = AsyncProcessExperienceCollector(
                            vec_env_builder=lambda: build_vec_env(env_builder_args=env_builder_args,
                                                                log_folder=log_folder,
                                                                seed=seed,
                                                                num_envs=num_envs),
                            storage_torch_device=device,
                            buffer_size=hyperparams["train_freq"]*num_envs,
                            session=session)
    else:
        collector = AsyncThreadExperienceCollector( vec_env=build_vec_env(env_builder_args=env_builder_args,
                                                                log_folder=log_folder,
                                                                seed=seed,
                                                                num_envs=num_envs),
                                                    buffer_size=hyperparams["train_freq"]*num_envs,
                                                    storage_torch_device=device)
    observation_space = collector.observation_space()
    action_space = collector.action_space()
    collector.set_base_collector_model(lambda o,a: build_sac(o,a,hyperparams))
    model = build_sac(observation_space, action_space, hyperparams)

    # torchexplorer.watch(model, backend="wandb")
    wandb.watch((model, model._actor, model._q_net), log="all", log_freq=1000, log_graph=True)

    # compiled_model = th.compile(model)

    rb = ThDReplayBuffer(
        buffer_size=10_000_000,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        storage_torch_device=device,
        handle_timeout_termination=True,
        n_envs=num_envs)
    start_time = time.time()

    eval_env = half_cheetah_builder(log_folder=log_folder+"/eval",
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
                                  eval_freq_ep=10*num_envs,
                                  deterministic=False))
    callbacks.append(EvalCallback(eval_env=eval_env_rec_det,
                                  model=model,
                                  n_eval_episodes=1,
                                  eval_freq_ep=10*num_envs,
                                  deterministic=True))
    callbacks.append(CheckpointCallbackRB(save_path=log_folder+"/checkpoints",
                                          model=model,
                                          save_best=False,
                                          save_freq_ep=100*num_envs))
    try:
        train_off_policy(collector=collector,
            model = model,
            buffer = rb,
            total_timesteps=10_000_000,
            train_freq = hyperparams["train_freq"],
            learning_starts=500*num_envs*5,
            grad_steps=hyperparams["grad_steps"],
            batch_size=16384,
            log_freq=500,
            callbacks=callbacks)
    finally:
        collector.close()


def runFunction(seed, folderName, resumeModelFile, run_id, args):
    env_builder_args = {
        "forward_reward_weight" : 1.0,
        "ctrl_cost_weight" : 0.1,
        "reset_noise_scale" : 0.1,
        "exclude_current_positions_from_observation" : True,
        "video_save_freq" : 0,
        "max_episode_steps" : 1000, # about 50Hz
        }

    hyperparams = {"train_freq" : 50,
                   "grad_steps" : 25,
                   "q_lr" : 0.005,
                   "policy_lr" : 0.0005}
    main(seed, folderName, run_id, args, env_builder_args, hyperparams)

if __name__ == "__main__":

    import os
    import argparse
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=1,
                launchFilePath=__file__,
                resumeFolder = None,
                args = args,
                debug_level = -10,
                start_adarl=False,
                pkgs_to_save=["adarl","rreal"])
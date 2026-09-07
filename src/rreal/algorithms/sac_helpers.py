#!/usr/bin/env python3  

from __future__ import annotations

from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.envs.GymToLr import GymToLr
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
from adarl.envs.lr_wrappers.ObsToDict import ObsToDict
from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface
from adarl.envs.vec.Runner2VecGymWrapper import Runner2VecGymWrapper
from adarl.envs.vector_env_checker import VectorEnvChecker
from adarl.envs.vector_env_logger import VectorEnvLogger
from adarl.utils.ThDictEpReplayBuffer import ThDictEpReplayBuffer
from adarl.utils.ThVecDictEpReplayBuffer import ThVecDictEpReplayBuffer
from adarl.utils.async_vector_env import AsyncVectorEnvShmem
from adarl.utils.buffers import ThDReplayBuffer
from rreal.utils.RNDNoveltyEstimator import RNDEstimatorHyperparams, SAC_RND_reward_hyperparams
from rreal.utils.callbacks import EvalCallback, CheckpointCallbackRB
from rreal.algorithms.collectors import AsyncProcessExperienceCollector, AsyncThreadExperienceCollector, SyncExperienceCollector
from rreal.algorithms.rl_agent import RLAgent
from rreal.algorithms.sac import SAC, train_off_policy, SAC_init_hparams, TransitionAugmentorFunction
from rreal.feature_extractors import get_feature_extractor
from rreal.feature_extractors.mixed_feature_extractor import MixedFeatureExtractorInitArgs
from rreal.tmp.gym_transform_observation import DtypeObservation
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.session
import adarl.utils.session as session
import adarl.utils.spaces as spaces
import gymnasium as gym
import inspect
import numpy as np
import os
import time
import torch as th
import typing
import wandb 
import math
from dataclasses import dataclass
from rreal.utils.utils import filter_dict_space
from typing import Any

class EnvBuilderProtocol(typing.Protocol):
    def __call__(self, seed : int, log_folder : str, is_eval : bool, env_builder_args : dict) -> tuple[gym.Env,float]:
        ...

class VecEnvBuilderProtocol(typing.Protocol):
    def __call__(self, seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = "") -> gym.vector.VectorEnv:
        ...

class VecEnvRunnerBuilderProtocol(typing.Protocol):
    def __call__(self, seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = "", autoreset = True, quiet = False) -> EnvRunnerInterface:
        ...

class TargetEntropyAnnealer:
    def __init__(self, start_target: float = -1.5,
                 end_target: float = -5.0,
                 start_reference_threshold: float = 0.25,
                 end_reference_threshold: float = 0.0,
                 reference_smoothing_alpha: float = 0.999,
                 reference_key: str = "linvel_q95"):
        self._start_target = start_target
        self._end_target = end_target
        self._start_reference_threshold = start_reference_threshold
        self._end_reference_threshold = end_reference_threshold
        self._reference_key = reference_key

        self._reference_smoothing_alpha = reference_smoothing_alpha
        self._smoothed_reference : float | None = None

    def anneal(self, global_exp_step : int, train_iterations : int) -> float:
        import adarl.utils.session
        import adarl.utils.dbg.ggLog as ggLog
        linvelq95 = adarl.utils.session.default_session.run_info["extras"].get(self._reference_key, None)

        if self._smoothed_reference is None:
            self._smoothed_reference = linvelq95
        else:
            a = self._reference_smoothing_alpha
            self._smoothed_reference = a*self._smoothed_reference + (1.0 - a)*linvelq95
        linvelq95 = self._smoothed_reference

        if linvelq95 is None:
            ggLog.warn(f"target_entropy_annealing_linvelq95: No linvel q95 info found, using default target entropy factor of {self._start_target}")
            return self._start_target
        # print(f"Linvel q95 = {linvelq95}")
        if linvelq95 < self._start_reference_threshold:
            w = self._start_reference_threshold - self._end_reference_threshold
            e = self._end_reference_threshold
            r = max(0.0, min(1.0, (linvelq95 - e)/w))
            return self._end_target + (self._start_target - self._end_target)*r
        else:
            return self._start_target

def gym_builder(seed, log_folder, is_eval, env_builder_args : dict[str,typing.Any]):
    # env = gym.make(env_builder_args["env_name"], render_mode="rgb_array")
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # return env, 0.02


    os.environ["MUJOCO_GL"]="egl"
    quiet = env_builder_args.pop("quiet",False)
    use_wandb = env_builder_args.pop("use_wandb",True)
    clip_action = env_builder_args.pop("clip_action",True)
    normalize_obs = env_builder_args.pop("normalize_obs",True)
    clip_obs = env_builder_args.pop("clip_obs",True)
    normalize_reward = env_builder_args.pop("normalize_reward",True)
    clip_reward = env_builder_args.pop("clip_reward",True)
    dict_obs = env_builder_args.pop("dict_obs",True)
    video_save_freq = env_builder_args.pop("video_save_freq",True)
    stepLength_sec = 0.05
    env = gym.make(env_builder_args["env_name"],
                    render_mode="rgb_array",
                    **env_builder_args["gym_args"])
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    if clip_action:
        env = gym.wrappers.ClipAction(env)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if clip_obs:
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    if clip_reward:
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env = DtypeObservation(env, dtype=np.float32)
    lrenv = GymToLr(env,
                    stepSimDuration_sec=stepLength_sec,
                    maxStepsPerEpisode=env_builder_args["max_episode_steps"],
                    copy_observations=True,
                    actions_to_numpy=True)
    if dict_obs:
        lrenv = ObsToDict(env=lrenv)
    lrenv.seed(seed=seed)
    env = GymEnvWrapper(env=lrenv,
                        episodeInfoLogFile = log_folder+f"/GymEnvWrapper_log.{seed:010d}.csv",
                        quiet=quiet,
                        use_wandb = use_wandb)
    if video_save_freq > 0:
        video_recorder_kwargs = {}
        env = RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq,
                                **video_recorder_kwargs)
    return env, 1/stepLength_sec


def build_vec_env(env_builder_args,
                  log_folder,
                  seed,
                  num_envs,
                  env_builder : EnvBuilderProtocol,
                  purely_numpy : bool = False,
                  logs_id = None,
                  collector_device : th.device = th.device("cuda"),
                  env_action_device : th.device|typing.Literal["numpy"] = th.device("cuda")) -> gym.vector.VectorEnv:
    if "use_wandb" not in env_builder_args:
        env_builder_args["use_wandb"] = False
    if logs_id is None:
        logs_id = session.default_session.run_info["run_id"]
    builders = [(lambda i: (lambda: env_builder(seed=seed+100000*i,
                                                log_folder=log_folder+f"/env_{i:003d}",
                                                is_eval = False,
                                                env_builder_args = env_builder_args)[0]
                                ))(i) for i in range(num_envs)]
    env = AsyncVectorEnvShmem(builders,
                               context="forkserver",
                               purely_numpy=purely_numpy,
                               shared_mem_device = collector_device,
                               copy_data=False,
                               worker_init_fn=session.set_current_session,
                               worker_init_kwargs={"session":session.default_session},
                                env_action_device = env_action_device)
    # env = VectorEnvLogger(env = env,
    #                        logs_id = logs_id)
    # env = VectorEnvChecker(env = env)
    return env


def env_builder2vec(env_builder : EnvBuilderProtocol, 
                    collector_device : th.device, 
                    env_action_device : th.device | typing.Literal["numpy"], 
                    purely_numpy : bool):
    return lambda seed, run_folder, num_envs, env_builder_args, env_name="": build_vec_env( env_builder=env_builder,
                                                                                            env_builder_args=env_builder_args,
                                                                                            log_folder=run_folder,
                                                                                            seed=seed,
                                                                                            num_envs=num_envs,
                                                                                            collector_device=collector_device,
                                                                                            env_action_device=env_action_device,
                                                                                            purely_numpy = purely_numpy)


def build_eval_callbacks(eval_configurations : list[dict],
                         vec_env_builder : VecEnvBuilderProtocol,
                         run_folder : str,
                         base_seed : int,
                         collector_device : th.device,
                         model : RLAgent):
    callbacks = []
    for eval_conf in eval_configurations:
        ggLog.info(f"Building eval config '{eval_conf['name']}'")
        eval_env = vec_env_builder(env_builder_args=eval_conf["env_builder_args"],
                                    run_folder=run_folder+f"/eval_"+eval_conf["name"],
                                    seed=base_seed+100000000,
                                    num_envs=eval_conf["num_envs"],
                                    env_name=eval_conf["name"])
        callbacks.append(EvalCallback(eval_env=eval_env,
                                    model=model,
                                    n_eval_episodes=eval_conf["eval_eps"],
                                    eval_freq_ep=eval_conf["eval_freq_ep"],
                                    deterministic=eval_conf["deterministic"],
                                    eval_name=eval_conf["name"],
                                    output_folder=run_folder+f"/eval_"+eval_conf["name"]+"/results",
                                    skip_first_eval=eval_conf.get("skip_first_eval", False)))
        ggLog.info(f"Built eval config '{eval_conf['name']}'")
    return callbacks

# def build_vec_env(env_builder, env_builder_args, log_folder, seed, num_envs) -> gym.vector.VectorEnv:
#     builders = [(lambda i: (lambda: env_builder(log_folder=log_folder,
#                                                   seed=seed+100000*i,
#                                                   env_builder_args = env_builder_args)
#                                 ))(i) for i in range(num_envs)]
#     envs = AsyncVectorEnvShmem(builders, context="forkserver", purely_numpy=False, shared_mem_device = th.device("cpu"), copy_data=False)
#     envs = VectorEnvLogger(env = envs)
#     return envs

def build_sac(obs_space : gym.Space, act_space : gym.Space, reward_space : gym.Space, hyperparams : SAC_init_hparams):
    ggLog.info(f"Building SAC agent with:\n"
               f"    obs_space: {obs_space}\n"
               f"    act_space: {act_space}\n"
               f"    reward_space: {reward_space}")
    agent = SAC(observation_space=obs_space,
                reward_space=reward_space,
                action_space=act_space,
                init_hparams=hyperparams)
    agent = th.compile(agent, mode="max-autotune", fullgraph=True)
    return agent


def build_sac_with_fe(obs_space : gym.Space,
                            act_space : gym.Space,
                            reward_space : gym.Space,
                            sac_hparams : SAC_init_hparams,
                            actor_feature_extractor_name : str | None,
                            critic_feature_extractor_name : str | None,
                            actor_fe_hparams : Any = None,
                            critic_fe_hparams : Any = None,
                            share_feature_extractor : bool | None = None) -> SAC:
    """Build a SAC agent, optionally with feature extractors selected by registered class name.

    Extractors are specified as a name plus their own hyperparameter object rather than as
    already-built instances, so that this builder stays picklable and can be shipped to the
    collector subprocesses. Passing only one of the two hparams reuses it for both roles.

    share_feature_extractor: True/False forces sharing on/off, None (the default) shares whenever
    both roles would build the very same extractor over the very same observations.
    """
    ggLog.info(f"Building SAC agent with:\n"
                   f"    obs_space: {obs_space}\n"
                   f"    act_space: {act_space}\n"
                   f"    reward_space: {reward_space}")
    actor_observation_space = filter_dict_space(obs_space, sac_hparams.actor_observation_filter)
    critic_observation_space = filter_dict_space(obs_space, sac_hparams.critic_observation_filter)

    # A single hparams object can be given and gets used for both roles
    if actor_fe_hparams is None:
        actor_fe_hparams = critic_fe_hparams
    if critic_fe_hparams is None:
        critic_fe_hparams = actor_fe_hparams
    for role, fe_name, fe_hparams in (("actor", actor_feature_extractor_name, actor_fe_hparams),
                                      ("critic", critic_feature_extractor_name, critic_fe_hparams)):
        if fe_name is None:
            continue
        if fe_hparams is None:
            raise AttributeError(f"{role}_feature_extractor_name is '{fe_name}' but no hyperparameters were "
                                 f"provided: set {role}_fe_hparams (or the other role's, which gets reused).")
        fe_device = getattr(fe_hparams, "device", None)
        if fe_device is not None and th.device(fe_device).type != th.device(sac_hparams.model_th_device).type:
            ggLog.warn(f"{role} feature extractor is configured for device {th.device(fe_device)}, but the "
                       f"model runs on {th.device(sac_hparams.model_th_device)}.")

    # SAC detects sharing by identity, so a shared extractor must be the very same object for both
    # roles. Sharing is only correct if the two roles are fed the same observations.
    can_share = (actor_feature_extractor_name == critic_feature_extractor_name and
                 actor_fe_hparams == critic_fe_hparams and
                 sac_hparams.actor_observation_filter == sac_hparams.critic_observation_filter)
    if share_feature_extractor is None:
        share_feature_extractor = can_share
    elif share_feature_extractor and not can_share:
        raise AttributeError(f"share_feature_extractor was requested, but actor and critic would not build the "
                             f"same extractor over the same observations: names "
                             f"'{actor_feature_extractor_name}'/'{critic_feature_extractor_name}', "
                             f"hparams match = {actor_fe_hparams == critic_fe_hparams}, filters "
                             f"{sac_hparams.actor_observation_filter}/{sac_hparams.critic_observation_filter}")

    if actor_feature_extractor_name is not None:
        act_fe = get_feature_extractor(actor_feature_extractor_name)( observation_space = actor_observation_space,
                                        hp = actor_fe_hparams)
    else:
        act_fe = None
    if critic_feature_extractor_name is None:
        critic_fe = None
    elif share_feature_extractor:
        critic_fe = act_fe
    else:
        critic_fe = get_feature_extractor(critic_feature_extractor_name)( observation_space = critic_observation_space,
                                        hp = critic_fe_hparams)
    ggLog.info(f"SAC feature extractors: actor = {type(act_fe).__name__}, critic = {type(critic_fe).__name__}"
               f"{' (shared: the same object as the actor one)' if act_fe is not None and critic_fe is act_fe else ''}")
    agent = SAC(observation_space=obs_space,
                reward_space=reward_space,
                action_space=act_space,
                init_hparams=sac_hparams,
                actor_feature_extractor=act_fe,
                critic_feature_extractor=critic_fe)
    agent = th.compile(agent, mode="max-autotune", fullgraph=True)
    return agent

def get_build_sac_with_fe_builder( sac_hparams : SAC_init_hparams,
                                        actor_feature_extractor_name : str | None,
                                        critic_feature_extractor_name : str | None,
                                        actor_fe_hparams : Any = None,
                                        critic_fe_hparams : Any = None,
                                        share_feature_extractor : bool | None = None):
    def builder(observation_space : gym.Space, action_space : gym.Space, reward_space : gym.Space):
        return build_sac_with_fe(obs_space=observation_space,
                                    act_space=action_space,
                                    reward_space=reward_space,
                                    sac_hparams=sac_hparams,
                                    actor_feature_extractor_name=actor_feature_extractor_name,
                                    critic_feature_extractor_name=critic_feature_extractor_name,
                                    actor_fe_hparams=actor_fe_hparams,
                                    critic_fe_hparams=critic_fe_hparams,
                                    share_feature_extractor=share_feature_extractor)
    return builder

def build_collector(use_processes : bool,
                    vec_env_builder : VecEnvBuilderProtocol,
                    env_builder_args : dict[str,typing.Any],
                    run_folder : str,
                    seed : int,
                    collector_device : th.device,
                    collector_buffer_size : int,
                    session : adarl.utils.session.Session,
                    num_envs : int,
                    deterministic_action_ratio : float = 0.0):
    vec_env_builder_norags = lambda: vec_env_builder(env_builder_args=env_builder_args,
                                                    run_folder=run_folder,
                                                    seed=seed,
                                                    num_envs=num_envs)
    if use_processes:
        collector = AsyncProcessExperienceCollector(
                            vec_env_builder=vec_env_builder_norags,
                            storage_torch_device=collector_device,
                            buffer_size=collector_buffer_size,
                            session=session,
                            deterministic_action_ratio=deterministic_action_ratio)
    else:
        collector = AsyncThreadExperienceCollector( vec_env=vec_env_builder_norags(),
                                                    buffer_size=collector_buffer_size,
                                                    storage_torch_device=collector_device)
    # collector = SyncExperienceCollector(vec_env=vec_env_builder_norags(),
    #                                     buffer_size=collector_buffer_size,
    #                                     storage_torch_device=collector_device)
    return collector

def wrap_with_logger(vec_env_builder : VecEnvBuilderProtocol,
                     max_obs_value : float = 255.0,
                     max_rew_value : float = 100.0) -> VecEnvBuilderProtocol:
    def wrapped_builder(seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = ""):
        # logs_id = session.default_session.run_info["run_id"]
        venv = vec_env_builder(seed = seed, run_folder = run_folder, num_envs = num_envs, env_builder_args = env_builder_args)
        venv = VectorEnvLogger(env = venv, logs_id = env_name, env_th_device=env_builder_args["th_device"], log_infos=env_builder_args["log_info_stats"])
        venv = VectorEnvChecker(env = venv, just_warn=False, max_obs_value=max_obs_value, max_rew_value=max_rew_value)
        return venv
    return wrapped_builder

def wrap_with_gym(vec_runner_builder : VecEnvRunnerBuilderProtocol) -> VecEnvBuilderProtocol:
    def wrapped_builder(seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = ""):
        return Runner2VecGymWrapper(  runner=vec_runner_builder( seed = seed,
                                                                run_folder = run_folder,
                                                                env_builder_args = env_builder_args,
                                                                num_envs = num_envs),
                                    quiet=env_builder_args["quiet"])
    return wrapped_builder


class AugmentorBuilder(typing.Protocol):
    def __call__(self, observation_space, action_space, reward_space) -> TransitionAugmentorFunction:
        ...

def sac_train(  seed : int,
                folderName : str,
                run_id : str,
                args,
                vec_env_builder : VecEnvBuilderProtocol | None,
                env_builder_args : dict,
                hyperparams : SAC_init_hparams,
                max_episode_duration : int,
                validation_buffer_size : int,
                validation_holdout_ratio : float,
                validation_batch_size : int,
                eval_configurations : list[dict] = [],
                checkpoint_freq : int = 100,
                collector_device : th.device | str | None = None,
                buffer_device : th.device | str | None = None,
                debug_level : int = 2,
                no_wandb : bool = False,
                log_weights_and_grads = False,
                transition_augmentor_builder: AugmentorBuilder | None = None,
                parallelize_collection : bool = True,
                actor_feature_extractor_name : str | None = None,
                critic_feature_extractor_name : str | None = None,
                actor_fe_hparams : Any = None,
                critic_fe_hparams : Any = None,
                share_feature_extractor : bool | None = None,
                use_rnd_exploration : bool = False,
                rnd_hyperparams : SAC_RND_reward_hyperparams | None = None):

    run_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                        inspect.currentframe(),
                                                        seed=seed,
                                                        run_id=run_id,
                                                        run_comment=args["comment"],
                                                        folderName=folderName,
                                                        debug=debug_level,
                                                        use_wandb=not no_wandb)
    validation_enabled = validation_buffer_size > 0 or validation_holdout_ratio > 0 or validation_batch_size > 0


    # if hyperparams.device == "cuda": hyperparams.device = "cuda:0"
    if isinstance(hyperparams.model_th_device, str):
        device = th.device(hyperparams.model_th_device)
    else:
        device = hyperparams.model_th_device
    if isinstance(buffer_device, str):
        buffer_device = th.device(buffer_device)
    if device.index is None:
        device = th.device(type=device.type, index=0)
    print(f"Device = {device}")
    if collector_device is None:
        collector_device = device
    if buffer_device is None:
        buffer_device = device

    if isinstance(collector_device, str):
        collector_device = th.device(collector_device)
    if isinstance(buffer_device, str):
        buffer_device = th.device(buffer_device) 
    if vec_env_builder is None:
        raise RuntimeError(f"You must specify either vec_env_builder")
    vec_env_builder = wrap_with_logger(vec_env_builder)
    # env setup
    collector = build_collector(use_processes = True,
                                vec_env_builder = vec_env_builder,
                                env_builder_args = env_builder_args,
                                run_folder = run_folder,
                                seed = seed,
                                num_envs=hyperparams.parallel_envs,
                                collector_device = collector_device,
                                collector_buffer_size = hyperparams.train_freq_vstep*hyperparams.parallel_envs,
                                session = session,
                                deterministic_action_ratio=hyperparams.deterministic_collection_ratio)
    if use_rnd_exploration:
        if rnd_hyperparams is None:
            rnd_hyperparams = SAC_RND_reward_hyperparams()
        rnd_hyperparams.scaler_hyperparams.rewards_num = spaces.get_1d_space_size(collector.reward_space())
        rnd_hyperparams.scaler_hyperparams.th_device = device
        rnd_hyperparams.estimator_hyperparams.th_device = device
        # Must be set before any agent is built: it decides the width of the reward vector, and
        # the collectors build their inference copies from these same hyperparams.
        hyperparams.rnd_hyperparams = rnd_hyperparams
    sac_builder = get_build_sac_with_fe_builder( sac_hparams = hyperparams,
                                                actor_feature_extractor_name = actor_feature_extractor_name,
                                                critic_feature_extractor_name = critic_feature_extractor_name,
                                                actor_fe_hparams = actor_fe_hparams,
                                                critic_fe_hparams = critic_fe_hparams,
                                                share_feature_extractor = share_feature_extractor)

    
    collector.set_base_collector_model(sac_builder)
    observation_space = collector.observation_space()
    action_space = collector.action_space()
    reward_space = collector.reward_space()

    model = sac_builder(observation_space, action_space, reward_space)

    # torchexplorer.watch(model, backend="wandb")
    if log_weights_and_grads:
        wandb.watch((model, model._actor, model._q_net), log="all", log_freq=1000, log_graph=False)

    if transition_augmentor_builder is not None:
        transition_augmentor = transition_augmentor_builder(observation_space, action_space, reward_space)
        model.set_transition_augmentor(transition_augmentor)

    if use_rnd_exploration:
        # Wired on the trained agent only, not inside the builder: the builder is also used for the
        # collectors' inference copies, which must not carry their own novelty estimator. Only the
        # reward space widening, set on the hyperparams above, is shared by both.
        assert rnd_hyperparams is not None
        from rreal.utils.RNDNoveltyEstimator import SAC_RND_reward_augmentor
        rnd_hyperparams.estimator_hyperparams.vec_input_size = (model.get_actor_encoding_size()
                                                                if rnd_hyperparams.use_actor_encoding
                                                                else model.get_critic_encoding_size())
        rnd_augmentor = SAC_RND_reward_augmentor(hyperparams=rnd_hyperparams)
        model.set_reward_augmentor_func(rnd_augmentor.get_augmented_rewards)
        model.register_postupdate_hook(rnd_augmentor.train_postupdate_hook)

    rewards_num = spaces.get_1d_space_size(reward_space)
    rb = ThVecDictEpReplayBuffer(buffer_size=hyperparams.buffer_size,
                                observation_space=observation_space,
                                action_space=action_space,
                                output_device=device,
                                storage_torch_device=buffer_device,
                                n_envs=hyperparams.parallel_envs,
                                max_episode_duration=max_episode_duration,
                                validation_buffer_size = validation_buffer_size,
                                validation_episodes=math.ceil(validation_holdout_ratio*hyperparams.parallel_envs),
                                min_episode_duration = 0,
                                disable_validation_set = True,
                                rewards_num=rewards_num)
    
    ggLog.info(f"Replay buffer occupies {rb.memory_size()/1024/1024:.2f}MB on {rb.storage_torch_device()}")
    
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

    ggLog.info(f"Starting training.")
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
            validation_batch_size=validation_batch_size,
            parallelize_experience_collection=parallelize_collection)
    finally:
        collector.close()

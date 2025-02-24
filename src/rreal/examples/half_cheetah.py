#!/usr/bin/env python3  

from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams, gym_builder
import copy



def runFunction(seed, folderName, resumeModelFile, run_id, args):
    import torch as th
    max_steps_per_episode = 1000
    num_envs = 8
    env_builder_args = {"env_name" : "HalfCheetah-v4",
                        "gym_args" : {  "forward_reward_weight" : 1.0,
                                        "ctrl_cost_weight" : 0.1,
                                        "reset_noise_scale" : 0.1,
                                        "exclude_current_positions_from_observation" : True},
                        "max_episode_steps" : max_steps_per_episode,
                        "quiet" : True,
                        "clip_action" : True,
                        "normalize_obs" : True,
                        "clip_obs" : True,
                        "normalize_reward" : True,
                        "clip_reward" : True,
                        "dict_obs" : True,
                        "video_save_freq" : -1,
                        "log_info_stats" : True,
                        "th_device" : th.device("cpu")}
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["video_save_freq"] = 1
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : num_envs*10,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "init_on_reset_ratio" : 1.0
    }
    eval_configs = [eval_conf_video_stoch]
    train_device = th.device("cuda")
    collect_device = th.device("cpu")
    if args["algo"].lower() == "sac":
        sac_train(seed, folderName, run_id, args,
                    env_builder=gym_builder,
                    env_builder_args = env_builder_args,
                    vec_env_builder=None,
                    hyperparams = SAC_hyperparams(  train_freq_vstep=16,
                                                    grad_steps=32,
                                                    parallel_envs = num_envs,
                                                    batch_size = 4096,
                                                    q_lr=1e-3,
                                                    policy_lr=3e-4,
                                                    device = train_device,
                                                    gamma = 0.99,
                                                    target_tau=0.005,
                                                    buffer_size=1_000_000,
                                                    total_steps = 10_000_000,
                                                    q_network_arch=[256,256],
                                                    policy_network_arch=[256,256],
                                                    learning_starts=20*num_envs*max_steps_per_episode,
                                                    log_freq_vstep = 1000,
                                                    reference_init_args={},
                                                    target_entropy=None),
                    collector_device=collect_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size = 0, #100_000,
                    validation_holdout_ratio = 0, #0.01,
                    validation_batch_size = 0,
                    eval_configurations=eval_configs) #256)
    elif args["algo"].lower() == "ppo":
        from rreal.algorithms.ppo import ppo_train, PPO_hyperparams
        raise RuntimeError(f"Use ppo2, this one is bugged")
        ppo_train(seed=seed,
                  folderName=folderName,
                  run_id=run_id,
                  args=args,
                  env_builder=gym_builder,
                  vec_env_builder=None,
                  env_builder_args=env_builder_args,
                  agent_hyperparams=PPO_hyperparams(minibatch_size=512,
                                                    th_device=train_device,
                                                    policy_arch=[64,64],
                                                    q_network_arch=[64,64],
                                                    total_steps = 1000_000,
                                                    q_lr=0.0003,
                                                    policy_lr=0.0003,
                                                    num_envs=num_envs,
                                                    num_steps=2048,
                                                    gamma=0.99,
                                                    update_epochs=10),
                  max_episode_duration=max_steps_per_episode,
                  validation_batch_size=0,
                  validation_buffer_size=0,
                  validation_holdout_ratio=0,
                  checkpoint_freq=-1,
                  collector_device=collect_device,
                  eval_configurations=eval_configs)
    elif args["algo"].lower() == "ppo2":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
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
                                                    log_freq_vstep=1000),
                max_episode_duration=1000,
                validation_batch_size=0,
                validation_buffer_size=0,
                validation_holdout_ratio=0,
                checkpoint_freq=-1,
                collector_device=th.device("cpu"),
                eval_configurations=eval_configs)




if __name__ == "__main__":

    import argparse
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--algo", required = False, type=str, help="Algorithm to use (SAC/PPO)")

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
#!/usr/bin/env python3  

from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams, env_builder2vec
from adarl.envs.examples.old.HopperEnv import HopperEnv
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
import typing

# def runFunction(seed, folderName, resumeModelFile, run_id, args):
#     import torch as th
#     sac_train(seed, folderName, run_id, args,
#                 env_builder=gym_builder,
#                 env_builder_args = {  "env_name" : "HalfCheetah-v4",
#                                     "forward_reward_weight" : 1.0,
#                                     "ctrl_cost_weight" : 0.1,
#                                     "reset_noise_scale" : 0.1,
#                                     "exclude_current_positions_from_observation" : True,
#                                     "max_episode_steps" : 1000},
#                 hyperparams = SAC_hyperparams(train_freq_vstep=1,
#                                     grad_steps=1,
#                                     q_lr=1e-3,
#                                     policy_lr=3e-4,
#                                     device = "cuda",
#                                     gamma = 0.99,
#                                     target_tau=0.005,
#                                     buffer_size=1_000_000,
#                                     total_steps = 10_000_000,
#                                     batch_size=256,
#                                     q_network_arch=[256,256],
#                                     policy_network_arch=[256,256],
#                                     learning_starts=1000,
#                                     parallel_envs = 16,
#                                     log_freq_vstep = 1000),
#                 collector_device=th.device("cuda"))
def builder(seed, log_folder, is_eval, env_builder_args : dict[str,typing.Any]):
    video_save_freq = env_builder_args.pop("video_save_freq")
    stepLength_sec = 0.05
    env = HopperEnv(simulatorController=PyBulletAdapter(enable_redering=video_save_freq>0,
                                                        stepLength_sec=stepLength_sec),
                    simulationBackend="bullet",
                    seed=seed)
    env = GymEnvWrapper(env, use_wandb=False)    
    if video_save_freq >0:
        env = RecorderGymWrapper(  env=env,
                                fps = 1/stepLength_sec,
                                outFolder=log_folder+"/videos/RecorderGymWrapper",
                                saveFrequency_ep=video_save_freq)
    return env, 1/stepLength_sec
    


def runFunction(seed, folderName, resumeModelFile, run_id, args):
    import torch as th
    max_steps_per_episode = 1000
    num_envs = 128
    sac_train(seed, folderName, run_id, args,
                env_builder=builder,
                vec_env_builder=env_builder2vec(builder, collector_device=th.device("cuda"), env_action_device=th.device("cuda"), purely_numpy=False),
                env_builder_args = {"video_save_freq" : 0},
                hyperparams = SAC_hyperparams(  train_freq_vstep=16,
                                                grad_steps=32,
                                                parallel_envs = num_envs,
                                                batch_size = 4096,
                                                q_lr=1e-3,
                                                policy_lr=3e-4,
                                                device = "cuda",
                                                gamma = 0.99,
                                                target_tau=0.005,
                                                buffer_size=1_000_000,
                                                total_steps = 10_000_000,
                                                q_network_arch=[256,256],
                                                policy_network_arch=[256,256],
                                                learning_starts=num_envs*max_steps_per_episode,
                                                log_freq_vstep = 1000,
                                                target_entropy_factor=-0.5,
                                                actor_log_std_init=-3.0,
                                                reference_init_args =   {}),
                collector_device=th.device("cpu"),
                max_episode_duration=max_steps_per_episode,
                validation_buffer_size = 100_000,
                validation_holdout_ratio = 0.01,
                validation_batch_size = 256,
                eval_configurations= [{   "name" : "video_stoch",
                                            "deterministic" : False,
                                            "eval_freq_ep" : 10*num_envs,
                                            "eval_eps" : 1,
                                            "env_builder_args" : {"video_save_freq":1},
                                            "num_envs" : 1
                                        }])



if __name__ == "__main__":

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
#!/usr/bin/env python3  

from rreal.examples.solve_sac import sac_train, SAC_hyperparams, gym_builder

def runFunction(seed, folderName, resumeModelFile, run_id, args):
    import torch as th
    sac_train(seed, folderName, run_id, args,
                env_builder=gym_builder,
                env_builder_args = {  "env_name" : "HalfCheetah-v4",
                                    "forward_reward_weight" : 1.0,
                                    "ctrl_cost_weight" : 0.1,
                                    "reset_noise_scale" : 0.1,
                                    "exclude_current_positions_from_observation" : True,
                                    "max_episode_steps" : 1000},
                hyperparams = SAC_hyperparams(train_freq_vstep=1,
                                    grad_steps=1,
                                    q_lr=1e-3,
                                    policy_lr=3e-4,
                                    device = "cuda",
                                    gamma = 0.99,
                                    target_tau=0.005,
                                    buffer_size=1_000_000,
                                    total_steps = 10_000_000,
                                    batch_size=256,
                                    q_network_arch=[256,256],
                                    policy_network_arch=[256,256],
                                    learning_starts=1000,
                                    parallel_envs = 16,
                                    log_freq_vstep = 1000),
                collector_device=th.device("cuda"))


# def runFunction(seed, folderName, resumeModelFile, run_id, args):
#     import torch as th
#     sac_train(seed, folderName, run_id, args,
#                 env_builder=gym_builder,
#                 env_builder_args = {"env_name" : "HalfCheetah-v4",
#                                     "forward_reward_weight" : 1.0,
#                                     "ctrl_cost_weight" : 0.1,
#                                     "reset_noise_scale" : 0.1,
#                                     "exclude_current_positions_from_observation" : True,
#                                     "max_episode_steps" : 1000,
#                                     "quiet" : True},
#                 hyperparams = SAC_hyperparams(  train_freq_vstep=16,
#                                                 grad_steps=16,
#                                                 parallel_envs = 256,
#                                                 batch_size = 4096,
#                                                 q_lr=1e-3,
#                                                 policy_lr=3e-4,
#                                                 device = "cuda",
#                                                 gamma = 0.99,
#                                                 target_tau=0.005,
#                                                 buffer_size=1_000_000,
#                                                 total_steps = 10_000_000,
#                                                 q_network_arch=[256,256],
#                                                 policy_network_arch=[256,256],
#                                                 learning_starts=1000,
#                                                 log_freq_vstep = 1000),
#                 collector_device=th.device("cpu"))



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
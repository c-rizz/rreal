#!/usr/bin/env python3

from rreal.examples.solve_sac import sac_train, SAC_hyperparams, gym_builder

def runFunction(seed, folderName, resumeModelFile, run_id, args):
    sac_train(seed, folderName, run_id, args,
              env_builder=gym_builder,
              env_builder_args = {  "env_name" : "InvertedPendulum-v4",
                                    "max_episode_steps" : 1000},
              hyperparams=SAC_hyperparams(train_freq=1,
                                  grad_steps=1,
                                  q_lr=0.005,
                                  policy_lr=0.0005,
                                  device = "cuda",
                                  gamma = 0.99,
                                  target_tau=0.005,
                                  buffer_size=1_000_000,
                                  total_steps = 10_000_000,
                                  batch_size=16384,
                                  q_network_arch=[64,64],
                                  policy_network_arch=[64,64],
                                  learning_starts=5000,
                                  parallel_envs = 1,
                                  log_freq_vstep = 1000,
                                  eval_freq_ep=100))

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
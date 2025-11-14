#!/usr/bin/env python3  

from rreal.examples.adarl_cartpole import cartpole_venv_builder
import adarl.utils.session
import inspect
import random
import numpy as np
import torch
from rreal.algorithms.sac_helpers import wrap_with_logger
from adarl.utils.utils import evaluatePolicyVec

def runFunction(seed, folderName, resumeModelFile, run_id, args):
    import torch as th
    max_steps_per_episode = 1000
    num_envs = 1
    env_builder_args = {"mode":args["mode"],
                        "th_device" : th.device("cuda") if args["mode"] == "mjx" else th.device("cpu"),
                        "enable_rendering" : False,
                        "log_info_stats" : True,
                        "quiet" : True,
                        "video_save_freq" : -1,
                        "max_steps" : max_steps_per_episode,
                        "step_length_sec" : 24/1024,
                        "task" : "balance",
                        "sparse_reward" :  True}
    device = th.device("cuda")
    vec_env_builder = cartpole_venv_builder
    
    run_folder, session = adarl.utils.session.adarl_startup(inspect.getframeinfo(inspect.currentframe().f_back)[0],
                                                    inspect.currentframe(),
                                                    seed=seed,
                                                    run_id=run_id,
                                                    run_comment=args["comment"],
                                                    folderName=folderName,
                                                    debug=0,
                                                    use_wandb=args.get("use_wandb", False))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # if hyperparams.device == "cuda": hyperparams.device = "cuda:0"
    if isinstance(device, str):
        device = th.device(device)
    if device.index is None:
        device = th.device(type=device.type, index=0)
    print(f"Device = {device}")
    vec_env_builder = wrap_with_logger(vec_env_builder)
    venv = vec_env_builder(env_builder_args=env_builder_args,
                           seed=seed,
                           run_folder=run_folder,
                           num_envs=num_envs)
    
    def policy(obs, deterministic=False):
        k = 5.0
        d = 0.1
        std = 0.01
        obs = obs["obs"]
        pole_angle = obs[:,2]
        pole_angular_vel = obs[:,3]
        perr = pole_angle
        verr = pole_angular_vel
        torque = -k*perr - d*verr
        if not deterministic:
            torque += th.empty_like(torque).normal_(0, std)
        act = torque/20
        return act.unsqueeze(-1), None

    results = evaluatePolicyVec(vec_env = venv,
                                model = None,
                                episodes = 10,
                                on_ep_done_callback = None,
                                predict_func = policy,
                                progress_bar = False,
                                images_return = None,
                                obs_return = None,
                                extra_info_stats = [],
                                deterministic = False)
    from pprint import pprint
    pprint(results)

    # 10_000 eps, num_envs=1000
    #     {'avg_pred_time': 8.991139487886921e-05,
    # 'avg_step_time': 0.037860333223788234,
    # 'collected_episodes': 40000,
    # 'collected_steps': 15323000,
    # 'fps': 17287.157145978184,
    # 'reward_mean': np.float32(376.89008),
    # 'reward_std': np.float32(203.67744),
    # 'steps_mean': np.float64(376.890075),
    # 'steps_std': np.float64(203.6774577401593),
    # 'success_ratio': np.float64(0.0)}
    # 500 eps, num_envs=1
    # {   'avg_pred_time': 5.798939661452498e-05,
    #     'avg_step_time': 0.009765546592488157,
    #     'collected_episodes': 500,
    #     'collected_steps': 196946,
    #     'fps': 101.36764714975715,
    #     'reward_mean': np.float32(393.892),
    #     'reward_std': np.float32(198.37553),
    #     'steps_mean': np.float64(393.892),
    #     'steps_std': np.float64(198.3755336123888),
    #     'success_ratio': np.float64(0.0)}
    # def p_welch(m1,m2,s1,s2,n1,n2):
    #     from scipy import stats
    #     import numpy as np
    #     # Welch t-statistic
    #     t = (m1 - m2) / np.sqrt(s1**2/n1 + s2**2/n2)
    #     # Welch–Satterthwaite degrees of freedom
    #     df = (s1**2/n1 + s2**2/n2)**2 / (
    #         (s1**4)/(n1**2*(n1-1)) + (s2**4)/(n2**2*(n2-1))
    #     )
    #     # two-sided p-value
    #     p = 2 * (1 - stats.t.cdf(abs(t), df))
    #     print("t =", t, "df =", df, "p =", p)


if __name__ == "__main__":

    import argparse
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")
    ap.add_argument("--algorithm", required = False, type=str, help="Algorithm to use (SAC/PPO)")
    ap.add_argument("--mode", required = False, type=str, help="Simulation mode to use")

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
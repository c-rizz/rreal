#!/usr/bin/env python3  

from rreal.algorithms.sac_helpers import sac_train, SAC_hyperparams, gym_builder, build_vec_env
import copy
from adarl.envs.examples.CartpoleVecEnv import CartpoleContinuousVecEnv
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
from adarl.envs.vec.GymVecRunnerWrapper import GymVecRunnerWrapper
from adarl.envs.vec.GymRunnerWrapper import GymRunnerWrapper
from adarl.envs.vec.lr_wrappers.ObsToDict import ObsToDict
from typing import Any


def cartpole_vrun_builder(  seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = "", autoreset = True, quiet = False):
    mode = env_builder_args["mode"]
    th_device = env_builder_args["th_device"]
    quiet = env_builder_args["quiet"]
    max_steps= env_builder_args["max_steps"]
    stepLength_sec= env_builder_args["step_length_sec"]
    video_save_freq = env_builder_args["video_save_freq"]
    if mode == "gz":
        raise NotImplementedError()
    elif mode == "gazebo":
        raise NotImplementedError()
    elif mode == "xbot":
        raise NotImplementedError()
    elif mode == "xbot-gazebo":
        from adarl_ros.adapters.RosXbotGazeboAdapter import RosXbotGazeboAdapter
        from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
        adapter = VecSimJointImpedanceAdapterWrapper(adapter = RosXbotGazeboAdapter(model_name = robot_name,
                                                                                    stepLength_sec = stepLength_sec,
                                                                                    forced_ros_master_uri = None,

                                                                                    maxObsDelay = float("+inf"),
                                                                                    blocking_observation = False,
                                                                                    is_floating_base = True,
                                                                                    reference_frame = "base_link",
                                                                                    torch_device = th_device,
                                                                                    fallback_cmd_stiffness = 200.0,
                                                                                    fallback_cmd_damping = 10.0,
                                                                                    allow_fallback = True,
                                                                                    jpos_cmd_max_vel = {},
                                                                                    jpos_cmd_max_vel_default = 10.0,
                                                                                    jpos_cmd_max_acc = {},
                                                                                    jpos_cmd_max_acc_default = 10.0),
                                                            vec_size = num_envs,
                                                            th_device = th_device)
    elif mode == "pybullet":
        from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
        from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
        adapter = VecSimJointImpedanceAdapterWrapper(adapter = PyBulletJointImpedanceAdapter(stepLength_sec=stepLength_sec,
                                                                            restore_on_reset=False,
                                                                            debug_gui=False,
                                                                            simulation_step=1/1024,
                                                                            enable_rendering=env_builder_args.pop("enable_rendering"),
                                                                            global_max_torque_position_control = 100,
                                                                            real_time_factor=None,
                                                                            th_device=th_device),
                                                            vec_size = num_envs,
                                                            th_device = th_device)
    elif mode == "mjx":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        import jax
        sim_dt = 4/1024
        adapter = MjxJointImpedanceAdapter( vec_size=num_envs,
                                            enable_rendering=env_builder_args.pop("enable_rendering"),
                                            jax_device=jax.devices("gpu")[0],
                                            output_th_device = th_device,
                                            sim_step_dt=sim_dt,
                                            step_length_sec=stepLength_sec,
                                            realtime_factor=-1.0,
                                            gui_env_index=0,
                                            default_max_joint_impedance_ctrl_torque=100.0,
                                            show_gui=False,
                                            log_freq=max_steps*(stepLength_sec/sim_dt),
                                            record_whole_joint_trajectories = False,
                                            log_freq_joints_trajectories = int(250*(50/1024)/(2/4096)),
                                            log_folder=run_folder,
                                            opt_preset="fastest",
                                            add_ground=False,
                                            add_sky=False)
    else:
        print(f"Requested unknown controller '{mode}'")
        exit(0)
    env = CartpoleContinuousVecEnv(adapter=adapter,
                                   maxStepsPerEpisode=max_steps,
                                   render=True,
                                   step_duration_sec=stepLength_sec,
                                   th_device=adapter.output_th_device(),
                                   img_obs=env_builder_args.pop("img_obs"),
                                   task=env_builder_args.pop("task"),
                                   sparse_reward=env_builder_args.pop("sparse_reward"),
                                   img_obs_resolution=env_builder_args.pop("img_obs_resolution"))
    env = ObsToDict(env=env)
    vrunner = EnvRunner(env=env, verbose=False, quiet=quiet, episodeInfoLogFile=run_folder+"/vec_runner.log",
                        render_envs=[0], autoreset=autoreset,
                        log_freq = max_steps)
    vrunner = EnvRunnerRecorderWrapper(vrunner,
                                    fps = 1/stepLength_sec,
                                    outFolder=run_folder+"/RunnerRecorder",
                                    env_index=0,
                                    saveFrequency_ep=video_save_freq,
                                    overlay_text_xy=(0.025,0.025),
                                    overlay_text_height=0.035,
                                    overlay_text_color_rgb=(255,150,0),
                                    overlay_text_func=lambda vo, a, r, te, tr, info:   
                                            f"\n"
                                            f"Angle    {info['pole_angle']: .3f}")
    return vrunner

def cartpole_venv_builder(  seed : int, run_folder : str, num_envs : int, env_builder_args : dict, env_name : str = ""):
    mode = env_builder_args["mode"]
    quiet = env_builder_args["quiet"]
    stepLength_sec= 50/1024
    if False: #mode == "pybullet":
        device = env_builder_args["th_device"]
        def single_env_builder(seed : int, log_folder : str, is_eval : bool, env_builder_args : dict[str, Any]):
            vrunner = cartpole_vrun_builder(seed = seed,
                                        run_folder = run_folder,
                                        env_builder_args = env_builder_args,
                                        num_envs = 1,
                                        autoreset = False,
                                        quiet = True)
            return GymRunnerWrapper(runner=vrunner, quiet=quiet), 1/stepLength_sec
        env = build_vec_env(env_builder=single_env_builder,
                            env_builder_args=env_builder_args,
                            log_folder=run_folder,
                            seed=seed,
                            num_envs=num_envs,
                            collector_device=device,
                            env_action_device = device)
    else:
        vrunner = cartpole_vrun_builder(seed = seed,
                                        run_folder = run_folder,
                                        env_builder_args = env_builder_args,
                                        num_envs = num_envs)
        env = GymVecRunnerWrapper(runner=vrunner, quiet=quiet)
    
    env.reset(seed=seed)
    return env



































def runFunction(seed, folderName, resumeModelFile, run_id, args):
    import torch as th
    max_steps_per_episode = 1000
    num_envs = 8
    env_builder_args = {"mode":args["mode"],
                        "th_device" : th.device("cuda") if args["mode"] == "mjx" else th.device("cpu"),
                        "enable_rendering" : False,
                        "log_info_stats" : True,
                        "quiet" : True,
                        "video_save_freq" : False}
    video_eval_env_builder_args = copy.deepcopy(env_builder_args)
    video_eval_env_builder_args["video_save_freq"] = 1
    video_eval_env_builder_args["enable_rendering"] = True
    eval_conf_video_stoch = {
        "name" : "video_stoch",
        "deterministic" : False,
        "eval_freq_ep" : num_envs*10,
        "eval_eps" : 1,
        "env_builder_args" : video_eval_env_builder_args,
        "num_envs" : 1,
        "init_on_reset_ratio" : 1.0
    }
    eval_configs = [
        eval_conf_video_stoch
        ]
    train_device = th.device("cuda",0)
    collect_device = th.device("cpu")
    if args["algorithm"].lower() == "sac":
        sac_train(seed, folderName, run_id, args,
                    env_builder=None,
                    env_builder_args = env_builder_args,
                    vec_env_builder=cartpole_venv_builder,
                    hyperparams = SAC_hyperparams(  train_freq_vstep=16,
                                                    grad_steps=16,
                                                    parallel_envs = num_envs,
                                                    batch_size = 512,
                                                    q_lr=1e-3,
                                                    policy_lr=3e-4,
                                                    device = train_device,
                                                    gamma = 0.99,
                                                    target_tau=0.005,
                                                    buffer_size=1_000_000,
                                                    total_steps = 10_000_000,
                                                    q_network_arch=[256,256],
                                                    policy_network_arch=[256,256],
                                                    learning_starts=5*max_steps_per_episode,
                                                    log_freq_vstep = 1000,
                                                    reference_init_args={},
                                                    target_entropy_factor=None),
                    collector_device=collect_device,
                    max_episode_duration=max_steps_per_episode,
                    validation_buffer_size = 0, #100_000,
                    validation_holdout_ratio = 0, #0.01,
                    validation_batch_size = 0,
                    eval_configurations=eval_configs) #256)
    elif args["algorithm"].lower() == "ppo":
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
    elif args["algorithm"].lower() == "ppo2":
        from rreal.algorithms.ppo2 import ppo_train, PPO_hyperparams
        ppo_train(  seed=seed,
                folderName=folderName,
                run_id=run_id,
                args=args,
                env_builder=None,
                vec_env_builder=cartpole_venv_builder,
                env_builder_args=env_builder_args,
                agent_hyperparams=PPO_hyperparams(  minibatch_size=512,
                                                    th_device=th.device("cuda"),
                                                    policy_arch=None,
                                                    q_network_arch=None,
                                                    q_lr=None,
                                                    policy_lr=3e-4,
                                                    update_epochs=10,
                                                    total_steps=1_000_000,
                                                    num_envs=num_envs,
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
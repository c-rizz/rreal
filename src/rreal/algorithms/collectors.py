
import os
import time

import gymnasium as gym
import numpy as np
import torch as th
from typing import List, Union, NamedTuple, Dict, Optional, Callable, Literal, Any
from adarl.utils.tensor_trees import map_tensor_tree, unstack_tensor_tree, stack_tensor_tree, is_all_finite
import adarl.utils.dbg.ggLog as ggLog
import copy
import threading
from adarl.utils.buffers import BasicStorage
import torch.multiprocessing as mp
from adarl.utils.shared_env_data import SimpleCommander
import ctypes
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
import gymnasium as gym
import adarl.utils.mp_helper as mp_helper
import adarl.utils.session as session
from adarl.utils.utils import pyTorch_makeDeterministic
from rreal.algorithms.rl_policy import RLPolicy
from abc import ABC, abstractmethod
import atexit
import setproctitle

class ExperienceCollector(ABC):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        buffer : Optional[BasicStorage] = None):
        self._vec_env = vec_env
        self._current_obs = None
        self._collector_model : th.nn.Module
        self._buffer = buffer
        self._vecenv_is_torch = True
        self._stats = { "t_act" : 0.,
                        "t_step" : 0.,
                        "t_copy" : 0.,
                        "t_final_obs" : 0.,
                        "t_add" : 0.}

    def set_base_collector_model(self, model_builder : Callable[[gym.spaces.Space, gym.spaces.Space],th.nn.Module]):
        self._collector_model = copy.deepcopy(model_builder(self.observation_space(), self.action_space()))

    def num_envs(self):
        return self._vec_env.num_envs

    def reset(self):
        if self._vec_env is not None:
            self._current_obs, info = self._vec_env.reset()
            
    @abstractmethod
    def observation_space(self) -> gym.Space:
        raise NotImplementedError()
    
    @abstractmethod
    def action_space(self) -> gym.Space:
        raise NotImplementedError()

    def collect_experience(self, policy, vsteps_to_collect, global_vstep_count, random_vsteps, policy_device,
                           buffer : BasicStorage):
        t0 = time.monotonic()
        with th.no_grad(): #just to be sure
            if  self._current_obs is None:
                raise RuntimeError(f"last_obs is not set. reset() should be called before running collect_experience the first time")
            num_envs = self.num_envs()
            t_act = 0.
            t_step = 0.
            t_copy = 0.
            t_final_obs = 0.
            t_add = 0.
            for step in range(vsteps_to_collect):
                t_pre_act = time.monotonic()
                obs = self._current_obs
                if global_vstep_count < random_vsteps:
                    actions = th.as_tensor(np.stack([self._vec_env.single_action_space.sample() for _ in range(num_envs)]))
                else:
                    th_obs = map_tensor_tree(obs, lambda a: th.as_tensor(a, device = policy_device))
                    # if not is_all_finite(th_obs):
                    #     raise RuntimeError(f"nonfinite values in th obs")
                    actions = policy.predict_action(th_obs)
                    if not self._vecenv_is_torch:
                        actions = actions.detach().cpu().numpy()
                t_post_act = time.monotonic()

                next_input_obss, rewards, terminations, truncations, infos = self._vec_env.step(actions)
                t_post_step = time.monotonic()

                # real_next_obs = copy.deepcopy(next_obs) #copy because we are going to modify it based on truncations
                t_post_copy = time.monotonic()
                # for idx, trunc in enumerate(truncations):
                #     if trunc:
                #         ggLog.info(f"Truncation happened")
                #         real_next_obs[idx] = infos["final_observation"][idx]
                # # truncated_envs = truncations.nonzero(as_tuple=True)
                # # map_tensor_tree(real_next_obs, lambda t : th.Tensor: t.index_copy_(dim=0,truncated_envs,))                

                # AsyncVectorEnvShmem always puts the consequent observation in final_observation.
                # If it wasn't available we could take it from final_observation/terminal_observation by masking with
                # (truncated or terminated)
                real_next_obss = infos["final_observation"] #stack_tensor_tree([info["real_next_observation"] for info in unstack_tensor_tree(infos)])
                t_post_final_obs = time.monotonic()
                buffer.add(obs=obs,
                            next_obs=real_next_obss,
                            action=actions,
                            reward=rewards,
                            terminated=terminations,
                            truncated=truncations)
                t_post_add = time.monotonic()
                # ggLog.info(f"added step {step} to buffer")
                self._current_obs = copy.deepcopy(next_input_obss) # copy it because it may get overwritten by the env during the next step
                # self._current_obs = next_obs
                obs, next_input_obss, rewards, terminations, truncations, infos = None, None, None, None, None, None # to avoid inadvertly using them

                t_act += t_post_act-t_pre_act
                t_step += t_post_step-t_post_act
                t_copy +=t_post_copy-t_post_step
                t_final_obs += t_post_final_obs-t_post_copy
                t_add += t_post_add-t_post_final_obs
            self._stats["t_act"] = t_act/vsteps_to_collect
            self._stats["t_step"] = t_step/vsteps_to_collect
            self._stats["t_copy"] = t_copy/vsteps_to_collect
            self._stats["t_final_obs"] = t_final_obs/vsteps_to_collect
            self._stats["t_add"] = t_add/vsteps_to_collect
            # ggLog.info(f"collection times = {t_act} {t_step} {t_copy} {t_final_obs} {t_add}")
        self._last_collection_wallduration = time.monotonic() - t0

    def collection_duration(self):
        return self._last_collection_wallduration


    def collect_experience_async(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        if self._collector_model is None or self._buffer is None:
            raise RuntimeError(f"Called collect_experience_async but base_model and buffer were not provided")
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._buffer.clear()
        self.collect_experience(self._collector_model,
                                vsteps_to_collect,
                                global_vstep_count,
                                random_vsteps,
                                self._collector_model.device,
                                self._buffer)
        
    def wait_collection(self, timeout = 0.0):
        if self._buffer is None:
            raise RuntimeError(f"Called collect_experience_async but buffer was not provided")
        return self._buffer
    
    def close(self):
        pass

class AsyncThreadExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        buffer_size : int,
                        storage_torch_device):
        super().__init__(vec_env=vec_env)

        self._start_collect = threading.Event()
        self._collect_done = threading.Event()
        self._collector_model : th.nn.Module
        self._running = True
        self._buffer_size = buffer_size
        self._storage_torch_device = storage_torch_device
        self._buffer = BasicStorage(buffer_size = self._buffer_size,
                                    observation_space=self._vec_env.single_observation_space,
                                    action_space=self._vec_env.single_action_space,
                                    n_envs=self._vec_env.num_envs,
                                    storage_torch_device=self._storage_torch_device,
                                    share_mem=True,
                                    allow_rollover=False)
        self._collector_thread = threading.Thread(target=self._worker, name="AsyncThreadExperienceCollector_worker")
        self._collector_thread.start()

    def _worker(self):
        while self._running and not session.default_session.is_shutting_down():
            got_set = self._start_collect.wait(timeout=2)
            if got_set:
                self._start_collect.clear()
                t0 = time.monotonic()
                self.collect_experience(policy=self._collector_model,
                                        vsteps_to_collect=self._vsteps_to_collect,
                                        global_vstep_count=self._global_vstep_count,
                                        random_vsteps=self._random_vsteps,
                                        policy_device=self._collector_model.device,
                                        buffer=self._buffer)
                self._last_collection_duration = time.monotonic() - t0
                self._collect_done.set()

    def collect_experience_async(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._vsteps_to_collect, self._global_vstep_count, self._random_vsteps = vsteps_to_collect, global_vstep_count, random_vsteps
        self._buffer.clear()
        self._start_collect.set()

    def wait_collection(self, timeout = 10.0):
        got_set = self._collect_done.wait(timeout=timeout)
        if not got_set:
            raise TimeoutError(f"Collector timed out waiting for collect (timeout = {timeout})")
        self._collect_done.clear()
        return self._buffer

    def close(self):
        self._running = False
        self._collector_thread.join()

    def observation_space(self):
        return self._vec_env.single_observation_space
    
    def action_space(self):
        return self._vec_env.single_action_space
        
    def collection_duration(self):
        return self._last_collection_duration
    

counter = 0
class AsyncProcessExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env_builder : Callable[[],gym.vector.VectorEnv],
                 buffer_size, storage_torch_device, start_method : Literal['fork', 'spawn', 'forkserver']= "forkserver",
                 session : session.Session = None,
                 seed : int = 0):
        super().__init__(vec_env=None)
        self._buffer_size = buffer_size
        self._storage_torch_device = storage_torch_device
        self._vec_env_builder = CloudpickleWrapper(vec_env_builder)
        self._state_dict : dict[str,th.Tensor]
        
        ctx = mp_helper.get_context(method=start_method)
        self._commander = SimpleCommander(mp_context=ctx, n_envs=1, timeout_s=60)
        self._collect_args = ctx.Array(ctypes.c_uint64, 3, lock = False)
        self._running = ctx.Value(ctypes.c_bool)
        self._running.value = ctypes.c_bool(True)
        self._last_collect_wall_duration = ctx.Value(ctypes.c_float)
        self._last_collect_wall_duration.value = 0.0
        p1, p2 = ctx.Pipe()
        global counter
        self._collector_process : mp.Process = ctx.Process(target = self._worker, args=(p2,session), name=f"async_experience_collector_{counter}")
        self._collector_process.start()
        self._pipe = p1
        self._seed = seed
        p2.close()
        counter += 1

        self._build_env()
        atexit.register(self.close)

    def _build_env(self):
        self._commander.set_command("build_env")
        self._obs_space : gym.Space
        self._action_space : gym.Space
        self._buffer, self._obs_space, self._action_space, self._num_envs = self._pipe.recv()
        self._commander.wait_done(timeout=60)

    def observation_space(self) -> gym.Space:
        return self._obs_space
    
    def action_space(self) -> gym.Space:
        return self._action_space

    def num_envs(self):
        return self._num_envs    

    def set_base_collector_model(self, model_builder : Callable[[gym.spaces.Space, gym.spaces.Space],th.nn.Module]):
        self._base_model_builder = CloudpickleWrapper(model_builder)
        self._pipe.send(self._base_model_builder)
        self._commander.set_command("build_model")
        self._commander.wait_done(timeout=60)
        self._state_dict = self._pipe.recv()

    def _worker(self, pipe, parent_session):
        ggLog.info(f"AsyncProcessExperienceCollector worker started with pid {os.getpid()}")
        setproctitle.setproctitle(mp.current_process().name)
        session.default_session = parent_session
        session.default_session.reapply_globals()
        self._pipe = pipe
        pyTorch_makeDeterministic(seed = session.default_session.run_info["seed"])
        while self._running.value:
            # ggLog.info(f"waiting command")
            cmd = self._commander.wait_command()
            # ggLog.info(f"got command {cmd}")
            if cmd == b"build_env":
                self._vec_env : gym.vector.VectorEnv = self._vec_env_builder.var()
                self.reset()
                self._buffer = BasicStorage(buffer_size = self._buffer_size,
                                            observation_space=self._vec_env.single_observation_space,
                                            action_space=self._vec_env.single_action_space,
                                            n_envs=self._vec_env.num_envs,
                                            storage_torch_device=self._storage_torch_device,
                                            share_mem=True,
                                            allow_rollover=False)
                self._obs_space = self._vec_env.single_observation_space
                self._action_space = self._vec_env.single_action_space
                self._num_envs = self._vec_env.num_envs
                self._pipe.send((self._buffer, 
                                 self._obs_space,
                                 self._action_space,
                                 self._num_envs))
            if cmd == b"build_model":
                # To ensure the correctly built model is used for collection we build it
                # directyl in the worker, to avoid any issue that may arise by sending it 
                # throucgh the pipe. Then we update its parameters by sharing the state dict
                self._base_model_builder = self._pipe.recv()
                self._collector_model = self._base_model_builder.var(self._obs_space, self._action_space)
                self._state_dict = self._collector_model.state_dict()
                self._pipe.send(self._state_dict)
            elif cmd == b"collect":
                vsteps_to_collect, global_vstep_count, random_vsteps = self._collect_args
                self._buffer.clear()
                t0 = time.monotonic()
                self.collect_experience(policy=self._collector_model,
                                        vsteps_to_collect=vsteps_to_collect,
                                        global_vstep_count=global_vstep_count,
                                        random_vsteps=random_vsteps,
                                        policy_device=self._collector_model.device,
                                        buffer = self._buffer)
                self._last_collect_wall_duration.value = time.monotonic() - t0
            elif cmd == b"close":
                ggLog.warn(f"{type(self)}: closing")
                self._vec_env.close()
                self._running.value = ctypes.c_bool(False)
            elif cmd is None:
                ggLog.warn(f"Worker timed out waiting for command. Will retry.")
            else:
                ggLog.warn(f"{type(self)}: Unexpected command {cmd}")
            if cmd is not None: # if a command was actually received
                self._commander.mark_done()
            # ggLog.info(f" {cmd} done")
        ggLog.info(f"Collector worker terminating")

    def collect_experience_async(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        for n,t in model_state_dict.items():
            self._state_dict[n].copy_(t)
        self._collect_args[:] = vsteps_to_collect, global_vstep_count, random_vsteps
        self._commander.set_command("collect")

    def wait_collection(self, timeout = 10.0):
        self._commander.wait_done(timeout=timeout)
        return self._buffer
    
    def collection_duration(self):
        return float(self._last_collect_wall_duration.value)
    
    def close(self):
        self._commander.set_command("close")
        self._collector_process.join(timeout=120)
        if self._collector_process.is_alive():
            self._collector_process.terminate()
            self._collector_process.join(timeout=30)
            if self._collector_process.is_alive():
                self._collector_process.kill()
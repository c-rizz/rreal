

from abc import ABC, abstractmethod
from adarl.utils.buffers import BasicStorage
from adarl.utils.shared_env_data import SimpleCommander
from adarl.utils.tensor_trees import map_tensor_tree, map2_tensor_tree
from adarl.utils.utils import pyTorch_makeDeterministic, dbg_check_finite
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from typing import List, Union, NamedTuple, Dict, Optional, Callable, Literal, Any
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.mp_helper as mp_helper
import adarl.utils.session as session
import atexit
import copy
import ctypes
import gymnasium as gym
import gymnasium as gym
import numpy as np
import os
import setproctitle
import threading
import time
import torch as th
import torch.multiprocessing as mp

class ExperienceCollector(ABC):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        buffer : Optional[BasicStorage] = None,
                        log_freq = 50):
        self._vec_env = vec_env
        self._current_obs : dict[str, th.Tensor] = None # type: ignore
        self._collector_model : th.nn.Module
        self._buffer = buffer
        self._vecenv_is_torch = True
        self._stats = { "t_act" : 0.,
                        "t_step" : 0.,
                        "t_copy" : 0.,
                        "t_final_obs" : 0.,
                        "t_add" : 0.,
                        "t_tot" : 0.}
        self._last_collection_end_wtime = time.monotonic()
        self._collect_count = 0
        self._log_freq = log_freq


    def set_base_collector_model(self, model_builder : Callable[[gym.spaces.Space, gym.spaces.Space],th.nn.Module]):
        self._collector_model = copy.deepcopy(model_builder(self.observation_space(), self.action_space()))

    def num_envs(self):
        return self._vec_env.unwrapped.num_envs

    def reset(self):
        if self._vec_env is not None:
            self._current_obs, info = self._vec_env.reset()
        self._current_obs = copy.deepcopy(self._current_obs) # make a copy of it to avoid inplace issues, this will be then in-place written during the steps
            
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
                # dbg_check_finite(self._current_obs)
                if global_vstep_count < random_vsteps:
                    actions = th.stack([th.as_tensor(self._vec_env.unwrapped.single_action_space.sample()) for _ in range(num_envs)])
                else:
                    th_obs = map_tensor_tree(self._current_obs, lambda a: th.as_tensor(a, device = policy_device))
                    # dbg_check_finite(th_obs)                    
                    actions = policy.predict_action(th_obs)
                    if not self._vecenv_is_torch:
                        actions = actions.detach().cpu().numpy()
                t_post_act = time.monotonic()

                next_input_obss, rewards, terminations, truncations, infos = self._vec_env.step(actions)
                t_post_step = time.monotonic()

                real_next_obss = infos["final_observation"] #stack_tensor_tree([info["real_next_observation"] for info in unstack_tensor_tree(infos)])
                t_post_final_obs = time.monotonic()

                # dbg_check_finite([self._current_obs, next_input_obss, rewards, terminations, truncations, actions])
                buffer.add(obs=self._current_obs,
                            next_obs=real_next_obss,
                            action=actions,
                            reward=rewards,
                            terminated=terminations,
                            truncated=truncations)
                t_post_add = time.monotonic()

                map2_tensor_tree(self._current_obs,next_input_obss, lambda l1, l2: l1.copy_(l2))
                t_post_copy = time.monotonic()

                next_input_obss, rewards, terminations, truncations, infos = None, None, None, None, None # to avoid inadvertly using them
                t_act += t_post_act-t_pre_act
                t_step += t_post_step-t_post_act
                t_copy +=t_post_copy-t_post_add
                t_final_obs += t_post_final_obs-t_post_step
                t_add += t_post_add-t_post_final_obs
        tf = time.monotonic()
        self._collect_count += 1
        t_tot = tf-t0
        self._stats["t_act"] = t_act
        self._stats["t_step"] = t_step
        self._stats["t_copy"] = t_copy
        self._stats["t_final_obs"] = t_final_obs
        self._stats["t_add"] = t_add
        self._stats["t_tot"] = t_tot
        self._stats["vsteps"] = vsteps_to_collect
        self._stats["vec_fps_ttot"] = vsteps_to_collect*self._vec_env.unwrapped.num_envs/t_tot
        self._stats["vec_fps_tstep"] = vsteps_to_collect*self._vec_env.unwrapped.num_envs/t_tot
        self._stats["ttot_wtime_ratio"] = t_tot/(tf-self._last_collection_end_wtime)
        self._last_collection_end_wtime = tf
        self._last_collection_wallduration = t_tot
        if self._collect_count % self._log_freq == 0:
            ggLog.info(f"collected: "+', '.join([f"{k}:{v:.6g}" for k,v in self._stats.items()]))

    def collection_duration(self):
        return self._last_collection_wallduration


    def start_collection(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        """Placeholder function, not actually asynchronous
        """
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
    
    def is_collecting(self):
        return False
    
    def close(self):
        pass

    def get_stats(self):
        return self._stats

class AsyncThreadExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        buffer_size : int,
                        storage_torch_device):
        super().__init__(vec_env=vec_env)

        self._start_collect = threading.Event()
        self._collect_done = threading.Event()
        self._collect_done.set() # As if we already collected something and it finished
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

    def start_collection(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._vsteps_to_collect, self._global_vstep_count, self._random_vsteps = vsteps_to_collect, global_vstep_count, random_vsteps
        self._buffer.clear()
        self._collect_done.clear()
        self._start_collect.set()

    def wait_collection(self, timeout = 10.0):
        got_set = self._collect_done.wait(timeout=timeout)
        if not got_set:
            raise TimeoutError(f"Collector timed out waiting for collect (timeout = {timeout})")
        return self._buffer

    def is_collecting(self):
        return not self._collect_done.is_set()

    def close(self):
        self._running = False
        self._collector_thread.join()

    def observation_space(self):
        return self._vec_env.single_observation_space
    
    def action_space(self):
        return self._vec_env.single_action_space
        
    def collection_duration(self):
        return self._last_collection_duration

class SyncExperienceCollector(ExperienceCollector):
    def __init__(self, vec_env : gym.vector.VectorEnv,
                        buffer_size : int,
                        storage_torch_device):
        super().__init__(vec_env=vec_env)

        self._collector_model : th.nn.Module
        self._buffer_size = buffer_size
        self._started_collect = False
        self._storage_torch_device = storage_torch_device
        self._buffer = BasicStorage(buffer_size = self._buffer_size,
                                    observation_space=self._vec_env.single_observation_space,
                                    action_space=self._vec_env.single_action_space,
                                    n_envs=self._vec_env.num_envs,
                                    storage_torch_device=self._storage_torch_device,
                                    share_mem=True,
                                    allow_rollover=False)

    def start_collection(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        self._collector_model.load_state_dict(model_state_dict, assign=False)
        self._vsteps_to_collect, self._global_vstep_count, self._random_vsteps = vsteps_to_collect, global_vstep_count, random_vsteps
        self._buffer.clear()
        self._started_collect = True

    def wait_collection(self, timeout = 10.0):
        if not self._started_collect:
            raise RuntimeError(f"You shold call collect_experience_async() before wait_collection()")
        t0 = time.monotonic()
        self.collect_experience(policy=self._collector_model,
                                vsteps_to_collect=self._vsteps_to_collect,
                                global_vstep_count=self._global_vstep_count,
                                random_vsteps=self._random_vsteps,
                                policy_device=self._collector_model.device,
                                buffer=self._buffer)
        self._last_collection_duration = time.monotonic() - t0        
        return self._buffer

    def is_collecting(self):
        return self._started_collect

    def close(self):
        self._running = False

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

        ctx = mp_helper.get_context(method=start_method)
        self._stats = mp_helper.get_manager(start_method).dict(self._stats)
        
        self._buffer_size = buffer_size
        self._storage_torch_device = storage_torch_device
        self._vec_env_builder = CloudpickleWrapper(vec_env_builder)
        self._state_dict : dict[str,th.Tensor]
        
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
        session.set_current_session(parent_session)
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
                                            observation_space=self._vec_env.unwrapped.single_observation_space,
                                            action_space=self._vec_env.unwrapped.single_action_space,
                                            n_envs=self._vec_env.unwrapped.num_envs,
                                            storage_torch_device=self._storage_torch_device,
                                            share_mem=True,
                                            allow_rollover=False)
                self._obs_space = self._vec_env.unwrapped.single_observation_space
                self._action_space = self._vec_env.unwrapped.single_action_space
                self._num_envs = self._vec_env.unwrapped.num_envs
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

    def start_collection(self, model_state_dict, vsteps_to_collect, global_vstep_count, random_vsteps):
        for n,t in model_state_dict.items():
            self._state_dict[n].copy_(t)
        self._collect_args[:] = vsteps_to_collect, global_vstep_count, random_vsteps
        self._commander.set_command("collect")

    def wait_collection(self, timeout = 10.0):
        self._commander.wait_done(timeout=timeout)
        return self._buffer
    
    def is_collecting(self):
        return self._commander.current_command() == "collect"

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
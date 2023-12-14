import numpy as np
import torch
import time
from rich import print
import supersuit as ss

from device_rl.device_handler import DeviceHandler
from device_rl.data import Data
from device_rl.module import Module
from device_rl.envs.simple.simple import SimpleEnv
from device_rl.envs.nvn.nvn import NvNEnv

from device_rl.abstract_sim.nvn import env as NvNAbstractEnv

def test_data_transfer():
    
    # handler = DeviceHandler()
    # data = Data(torch.from_numpy(np.random.rand(3, 3)))
    # data.to_device()
    # data.to_host()
    # print(data.get())

    return

def test_kernel():

    b = Data(263)
    b.to_device()

    kernel = Module('kernels/test.cu')
    kernel.load().set_config((3, 1), (3, 1, 1)).launch(b)

def test_simple_env():
    env = SimpleEnv()

    for i in range(10):
        env.step()
        env.render()
        time.sleep(0.5)

def test_nvn_env():

    n_envs = 9

    env = NvNEnv(
        n_envs = n_envs,
        n_agents = 32,
        n_opponents = 32,
        log_time = True
    )

    env.reset()
    for i in range(3000):
        # env.step()
        # print(env.state.get())
        env.sample()
        env.step()
        # print(env.action.get()[0, 0, :])
        # break
        env.render()
        time.sleep(.1)


def test_cuda_vs_abstract_sim():

    # purely sequential 

    n_trajectories = [2**n for n in range(17)]
    n_trajectories_small = [2**n for n in range(6)]
    # n_trajectories = [16]
    print(n_trajectories)

    env = NvNAbstractEnv()
    for n in n_trajectories_small:

        start = time.time()

        for _ in range(n):
            _, _ = env.reset()
            for _ in range(4001):
                action = {
                    'agent_0': env.action_space('agent_0').sample(),
                    'agent_1': env.action_space('agent_1').sample(),
                    'agent_2': env.action_space('agent_2').sample(),
                    'agent_3': env.action_space('agent_3').sample(),
                    'agent_4': env.action_space('agent_4').sample(),
                    'agent_5': env.action_space('agent_5').sample(),
                    'agent_6': env.action_space('agent_6').sample(),
                    'agent_7': env.action_space('agent_7').sample(),
                    'agent_8': env.action_space('agent_8').sample(),
                    'agent_9': env.action_space('agent_9').sample(),
                    'agent_10': env.action_space('agent_10').sample(),
                    'agent_11': env.action_space('agent_11').sample(),
                    'agent_12': env.action_space('agent_12').sample(),
                    'agent_13': env.action_space('agent_13').sample(),    
                    'agent_14': env.action_space('agent_14').sample(),    
                }
                _, _, _, _, _ = env.step(action)

        end = time.time()
        delta = (end - start) * 1000
        print(f"(sequential) Generated [cyan]{n}[/cyan] trajectories in [orange1]{delta}[/orange1] ms.")

        with open('tests/output2.txt', 'a') as file:
            file.write(str(delta) + '\n')
            file.close()


    env = NvNAbstractEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=16,
        num_cpus=8,
        base_class="stable_baselines3",
    )
    for n in n_trajectories_small:

        if n % 16 != 0: continue

        start = time.time()

        for _ in range(int(n / 16)):
            _ = env.reset()
            for _ in range(4001):
                action = env.action_space.sample()
                actions = [action for i in range(240)]
                _, _, _, _ = env.step(actions)

        end = time.time()
        delta = (end - start) * 1000
        print(f"(cpu-parallel) Generated [cyan]{n}[/cyan] trajectories in [orange1]{delta}[/orange1] ms.")

        with open('tests/output2.txt', 'a') as file:
            file.write(str(delta) + '\n')
            file.close()

        

    for n in n_trajectories:

        env = NvNEnv(
            n_envs = n,
            n_agents = 15,
            n_opponents = 15,
            log_time = False,
            render = False
        )

        start = time.time()


        env.reset()
        for i in range(4001):
            env.sample()
            env.step()

        end = time.time()
        delta = (end - start) * 1000
        print(f"(gpu-parallel) Generated [cyan]{n}[/cyan] trajectories in [orange1]{delta}[/orange1] ms.")

        with open('tests/output2.txt', 'a') as file:
            file.write(str(delta) + '\n')
            file.close()

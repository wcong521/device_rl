import numpy as np
import torch
import time

from device_rl.device_handler import DeviceHandler
from device_rl.data import Data
from device_rl.module import Module
from device_rl.envs.simple.simple import SimpleEnv
from device_rl.envs.nvn.nvn import NvNEnv

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

def text_nvn_env():
    env = NvNEnv(
        num_envs = 1024,
        num_agents = 32,
    )
    env.reset()
    print(env.agents.get())

    




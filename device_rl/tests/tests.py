import numpy as np
import torch

from device_rl.device_handler import DeviceHandler
from device_rl.data import Data
from device_rl.kernel import Kernel

def test_data_transfer():
    
    handler = DeviceHandler()
    data = Data(torch.from_numpy(np.random.rand(3, 3)))
    data.to_device()
    data.to_host()
    print(data.get())


def test_kernel():

    b = Data(263)
    b.to_device()

    kernel = Kernel('kernels/test.cu')
    kernel.load().set_config((3, 1), (3, 1, 1)).launch(b)




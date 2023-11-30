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

    arr = Data(torch.from_numpy(np.zeros((3, 4))))
    n = Data(torch.tensor(9, dtype=torch.float32))
    arr.to_device()
    n.to_device()

    kernel = Kernel('kernels/test.cu')
    kernel.load()
    kernel.run([(3, 1), (3, 1, 1)], [arr, n])

    print(arr.get())




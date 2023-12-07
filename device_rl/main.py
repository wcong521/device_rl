import matplotlib.pyplot as plt
import torch
import pycuda.driver as cuda


from device_rl.tests.tests import test_data_transfer, test_kernel, test_simple_env, text_nvn_env

# https://stackoverflow.com/questions/2816992/double-precision-floating-point-in-cuda
torch.set_default_dtype(torch.float32)

plt.style.use('dark_background')
cuda.init()
context = cuda.Device(0).make_context()




text_nvn_env()




context.pop()
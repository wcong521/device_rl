import matplotlib.pyplot as plt
import torch
import pycuda.driver as cuda


from device_rl.tests.tests import test_nvn_env, test_cuda_vs_abstract_sim

# https://stackoverflow.com/questions/2816992/double-precision-floating-point-in-cuda
torch.set_default_dtype(torch.float32)

plt.style.use('dark_background')
cuda.init()
context = cuda.Device(0).make_context()



# test_nvn_env()
test_cuda_vs_abstract_sim()




context.pop()
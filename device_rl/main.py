import torch
import pycuda.driver as cuda


from device_rl.tests.tests import test_data_transfer, test_kernel

# https://stackoverflow.com/questions/2816992/double-precision-floating-point-in-cuda
torch.set_default_dtype(torch.float32)

cuda.init()
context = cuda.Device(0).make_context()

test_kernel()

context.pop()
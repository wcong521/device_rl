import uuid

import torch
import pycuda.driver as cuda

# class CudaTensorHolder(cuda.PointerHolderBase):
#     """
#     A class that facilitates casting tensors to pointers.
#     """

#     def __init__(self, t):
#         super().__init__()
#         self.gpudata = t.data_ptr()

class Data:
    def __init__(self, data):
        self.id = uuid.uuid4()
        self.data = data

        # https://stackoverflow.com/questions/2816992/double-precision-floating-point-in-cuda
        if self.data.dtype == torch.float64:
            self.data = self.data.type(torch.FloatTensor)

    def where(self):
        return 'host' if self.data.get_device() == -1 else 'device'
    
    def to_device(self):
        if self.where() == 'device':
            print('Already on the device.')
            return
        
        self.data = self.data.cuda()
        # self.pointer = self.data.data_ptr()

    def to_host(self):
        if self.where() == 'host':
            print('Already on the host.')
            return
        
        self.data = self.data.cpu()

    def get(self):
        return self.data

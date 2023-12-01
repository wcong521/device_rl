import uuid

import numpy as np
import torch
import pycuda.driver as cuda

class Data:
    def __init__(self, data):
        self.id = uuid.uuid4()
        self.is_scalar = False

        # https://stackoverflow.com/questions/2816992/double-precision-floating-point-in-cuda
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                self.data = torch.from_numpy(data).type(torch.float32)
            elif data.dtype == np.int64:
                self.data = torch.from_numpy(data).type(torch.int32)
            else:
                raise Exception(f'Unsupported array type: {data.dtype}.')
        else:
            self.is_scalar = True
            if isinstance(data, float):
                self.data = np.float32(data)
            elif isinstance(data, int) and not isinstance(data, bool):
                self.data = np.int32(data)
            else:
                raise Exception(f'Unsupported scalar type: {type(data)}.')
            

    def where(self):
        return 'host' if self.is_scalar or self.data.get_device() == -1 else 'device'
    
    def to_device(self):
        if self.where() == 'device':
            print('Already on the device.')
            return
        
        if not self.is_scalar:
            self.data = self.data.cuda()

    def to_host(self):
        if self.where() == 'host':
            print('Already on the host.')
            return
        
        if not self.is_scalar:
            self.data = self.data.cpu()

    def get(self):
        return self.data

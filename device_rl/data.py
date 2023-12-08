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
            if np.issubdtype(data.dtype, np.floating):
                self._data = torch.from_numpy(data).type(torch.float32)
            elif np.issubdtype(data.dtype, np.integer):
                self._data = torch.from_numpy(data).type(torch.int32)
            elif np.issubdtype(data.dtype, np.bool_):
                self._data = torch.from_numpy(data).type(torch.uint8)
            else:
                raise Exception(f'Unsupported array type: {data.dtype}.')
        else:
            self.is_scalar = True
            if isinstance(data, float):
                self._data = np.float32(data)
            elif isinstance(data, int) and not isinstance(data, bool):
                self._data = np.int32(data)
            # elif isinstance(data, bool):
            #     self._data = np.uint8(data)
            else:
                raise Exception(f'Unsupported scalar type: {type(data)}.')
            

    def where(self):
        return 'host' if self.is_scalar or self._data.get_device() == -1 else 'device'
    
    def to_device(self):
        if self.where() == 'device':
            print('Already on the device.')
            return
        
        if not self.is_scalar:
            self._data = self._data.cuda()

    def to_host(self):
        if self.where() == 'host':
            print('Already on the host.')
            return
        
        if not self.is_scalar:
            self._data = self._data.cpu()

    def copy_to_host(self):
        if self.is_scalar:
            return self._data
        else: 
            return self._data.clone() if self.where() == 'host' else self._data.clone().cpu()

    def get(self):
        return self._data

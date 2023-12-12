import os
import subprocess

import pycuda.driver as cuda
import pycuda.compiler as compiler
import numpy as np

from device_rl.data import Data

bin_dir_path = 'bin'

class Module:

    def __init__ (self, path):
        self.cu_path = path
        return
    
    def load(self):
        bin_path = f'{bin_dir_path}/test.fatbin'

        # compile

        with subprocess.Popen(
            f'mkdir -p {bin_dir_path}', 
            shell=True, 
            stderr=subprocess.STDOUT
        ) as process:
            if process.wait() != 0:
                raise Exception(f"Failed to create bin directory at {bin_dir_path}")
        print(f"Created bin directory at {bin_dir_path}.")

        if os.path.exists(f'{bin_path}'):
            os.remove(f'{bin_path}')

        try:
            cc = cuda.Device(0).compute_capability() 
            with subprocess.Popen(
                f"nvcc --fatbin -arch=sm_{cc[0]}{cc[1]} {self.cu_path} -o {bin_path}", 
                shell=True, 
                stderr=subprocess.STDOUT
            ) as process:
                if process.wait() != 0: raise Exception()
        except Exception as err:
            print(err)

        self._module = cuda.module_from_file(bin_path)
        return self
    

    def launch(self, name, grid, block, shared=0):
        if not grid:
            raise Exception('Grid dimensions not specified in kernel launch.')
        
        if not block:
            raise Exception('Block dimensions not specified in kernel launch.')
        
        kernel = self._module.get_function(name)
        return lambda *args: kernel(*[a.get() for a in args], grid=grid, block=block, shared=shared)



        


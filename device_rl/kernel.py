import os
import subprocess

import pycuda.driver as cuda
import pycuda.compiler as compiler
import numpy as np

from device_rl.data import Data

bin_dir_path = 'bin'

class Kernel:

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

        self.module = cuda.module_from_file(bin_path)
        self.function = self.module.get_function('test')

        return self
    

    def set_config(self, grid, block):
        self.grid = grid
        self.block = block

        return self

    def launch(self, *args):
        self.function(*[a.data for a in args], grid=self.grid, block=self.block)



        


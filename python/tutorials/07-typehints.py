import time
import os
import sys
import importlib
import textwrap

from typing import Tuple
import torch

import triton

def define_empty_kernel(file_path, num_tensor_args):
    arg_str = ",".join([f"arg{i}: torch.Tensor" for i in range(40)])
    arg_str += ", n_elements: int, BLOCK_SIZE: tl.constexpr"
    func_str = f"""
    import torch

    import triton
    import triton.language as tl

    @triton.jit
    def empty_kernel({arg_str}):
        pass
    """
    with open(file_path, "w") as f:
        f.write(textwrap.dedent(func_str))

def import_empty_kernel(file_path):
    directory, filename = os.path.split(file_path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, directory)

    module = importlib.import_module(module_name)
    empty_kernel = module.empty_kernel
    return empty_kernel

def empty(*kernel_args: Tuple[torch.Tensor]):
    first_arg = kernel_args[0]
    n_elements = first_arg.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    empty_kernel[grid](*kernel_args, n_elements, BLOCK_SIZE=1024)
    # torch.cuda.synchronize()
    # start_time = time.time()
    # empty_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, device=torch.cuda.current_device())
    # end_time = time.time()
    # print(f"Total time = {(end_time - start_time) * 1e6} usec")

file_path = '/tmp/empty_kernel.py'
define_empty_kernel(file_path, 40)
empty_kernel = import_empty_kernel(file_path)

torch.manual_seed(0)
size = 98432
kernel_args = (torch.rand(size, device='cuda') for i in range(40))
output_triton = empty(*kernel_args)
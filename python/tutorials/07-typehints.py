import torch

import triton
import time
import triton.language as tl


@triton.jit
def empty_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    pass


def empty_func(x: torch.Tensor, y: torch.Tensor):
    print("Running empty kernel")
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    start_time = time.time()
    empty_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    end_time = time.time()

    latency = end_time - start_time
    print("Kernel launch latency: {:.6f} seconds".format(latency))


torch.manual_seed(0)
size = 1024
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_triton = empty_func(x, y)
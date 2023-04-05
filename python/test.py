import triton
import pytest
import torch
import triton.language as tl
import numpy as np
from numpy.random import RandomState


@pytest.mark.parametrize("M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis",
                         [(*shape_nw, 'softmax', allow_tf32, in_dtype, out_dtype, axis)
                          for shape_nw in [[128, 16, 16, 4]]
                          for allow_tf32 in [True]
                          for in_dtype, out_dtype in [('float32', 'float32')]
                          for axis in [0, 1]])
def test_dot(M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis, device='cuda'):
    capability = torch.cuda.get_device_capability()
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               out_dtype: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               AXIS: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32, out_dtype=out_dtype)
        max = tl.max(z, AXIS)
        if AXIS == 1:
            z = z - max[:, None]
        else:
            z = z - max[None, :]
        min = tl.min(z, AXIS)
        if AXIS == 1:
            z = z - min[:, None]
        else:
            z = z - min[None, :]
        w = tl.load(Ws)
        z = tl.dot(z.to(w.dtype), w, out_dtype=out_dtype)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    x = rs.randint(0, 4, (M, K)).astype(in_dtype)
    y = rs.randint(0, 4, (K, N)).astype(in_dtype)
    w = np.ones((N, N)).astype(in_dtype)
    if in_dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    w_tri = torch.tensor(w, device=device)
    z = 1 + rs.randint(0, 4, (M, N)).astype(in_dtype)

    z_tri = torch.tensor(z, device=device)
    out_dtype = tl.float32

    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         out_dtype,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         AXIS=axis,
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps)
    z_ref = np.matmul(x, y)
    z_ref = z_ref - np.max(z_ref, axis=axis, keepdims=True)
    z_ref = z_ref - np.min(z_ref, axis=axis, keepdims=True)
    z_ref = np.matmul(z_ref, w)
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    if in_dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
    elif out_dtype == tl.float16:
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
    else:
        np.savetxt('np.out', z_ref)
        np.savetxt('triton.out', z_tri.cpu().numpy())
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01)

@pytest.mark.parametrize("M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis",
                         [(*shape_nw, 'softmax', allow_tf32, in_dtype, out_dtype, axis)
                          for shape_nw in [[128, 16, 16, 4]]
                          for allow_tf32 in [True]
                          for in_dtype, out_dtype in [('float32', 'float32')]
                          for axis in [0]])
def test_reduce(M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis, device='cuda'):
    capability = torch.cuda.get_device_capability()

    # triton kernel
    @triton.jit
    def reduce_kernel(X, Z, stride_xm, stride_xn, stride_zm, stride_zn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_n * stride_zn
        x = tl.load(Xs)
        z = tl.max(x, AXIS)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    inc = [[row*N + col for col in range(N)] for row in range(M)]
    x = np.array(inc).astype(in_dtype)
    # x = rs.randint(0, 4, (M, N)).astype(in_dtype)
    x_tri = torch.tensor(x, device=device)
    z = 1 + rs.randint(0, 4, (1, N)).astype(in_dtype)
    z_tri = torch.tensor(z, device=device)

    pgm = reduce_kernel[(1, 1)](x_tri, z_tri, x_tri.stride(0), x_tri.stride(1), z_tri.stride(0), z_tri.stride(1), M, N, axis)
    z_ref = x
    z_ref = np.max(z_ref, axis=axis, keepdims=True)

    # compare
    # print(z_ref[:,0], z_tri[:,0])
    # XXX: Somehow there's a larger difference when we use float32
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
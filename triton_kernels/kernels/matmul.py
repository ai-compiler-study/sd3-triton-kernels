import torch
import triton
import triton.language as tl


@triton.jit
def matmul_i8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    bm = tl.program_id(0)
    bn = tl.program_id(1)
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for bk in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * K + (bk + offsets_k[None, :])
        b_ptrs = b_ptr + (bk + offsets_k[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
        acc += c
    c_ptrs = c_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(c_ptrs, acc)


def matmul_i8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication for int8 tensors using Triton.

    Args:
        a (torch.Tensor): Input tensor of shape (M, K) with dtype int8.
        b (torch.Tensor): Input tensor of shape (K, N) with dtype int8.

    Returns:
        torch.Tensor: Resulting tensor of shape (M, N) with dtype int32.
    """
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(b.shape[0], -1)
    M, K = a.shape
    K2, N = b.shape
    assert a.dtype == torch.int8, "Input tensor a must be of dtype int8"
    assert b.dtype == torch.int8, "Input tensor b must be of dtype int8"
    assert K == K2, "Inner dimensions must match"

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32

    c = torch.empty((M, N), dtype=torch.int32, device=a.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_i8_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    c = c.reshape(*a.shape[:-1], -1)
    return c


@triton.jit
def matmul_f32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    bm = tl.program_id(0)
    bn = tl.program_id(1)
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for bk in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * K + (bk + offsets_k[None, :])
        b_ptrs = b_ptr + (bk + offsets_k[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
        acc += c
    c_ptrs = c_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(c_ptrs, acc)


def matmul_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication for float32 tensors using Triton.

    Args:
        a (torch.Tensor): Input tensor of shape (M, K) with dtype float32.
        b (torch.Tensor): Input tensor of shape (K, N) with dtype float32.

    Returns:
        torch.Tensor: Resulting tensor of shape (M, N) with dtype float32.
    """
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(b.shape[0], -1)
    M, K = a.shape
    K2, N = b.shape
    assert a.dtype == torch.float32, "Input tensor a must be of dtype float32"
    assert b.dtype == torch.float32, "Input tensor b must be of dtype float32"
    assert K == K2, "Inner dimensions must match"

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_f32_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    c = c.reshape(*a.shape[:-1], -1)
    return c


@triton.jit
def matmul_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    bm = tl.program_id(0)
    bn = tl.program_id(1)
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for bk in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * K + (bk + offsets_k[None, :])
        b_ptrs = b_ptr + (bk + offsets_k[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
        acc += c
    c_ptrs = c_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(c_ptrs, acc)


def matmul_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication for bfloat16 tensors using Triton.

    Args:
        a (torch.Tensor): Input tensor of shape (M, K) with dtype bfloat16.
        b (torch.Tensor): Input tensor of shape (K, N) with dtype bfloat16.

    Returns:
        torch.Tensor: Resulting tensor of shape (M, N) with dtype float32.
    """
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(b.shape[0], -1)
    M, K = a.shape
    K2, N = b.shape
    assert a.dtype == torch.bfloat16, "Input tensor a must be of dtype bfloat16"
    assert b.dtype == torch.bfloat16, "Input tensor b must be of dtype bfloat16"
    assert K == K2, "Inner dimensions must match"

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_bf16_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    c = c.reshape(*a.shape[:-1], -1)
    return c


@triton.jit
def matmul_f16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    bm = tl.program_id(0)
    bn = tl.program_id(1)
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for bk in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * K + (bk + offsets_k[None, :])
        b_ptrs = b_ptr + (bk + offsets_k[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
        acc += c
    c_ptrs = c_ptr + (bm * BLOCK_SIZE_M + offsets_m[:, None]) * N + (bn * BLOCK_SIZE_N + offsets_n[None, :])
    tl.store(c_ptrs, acc)


def matmul_f16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication for float16 tensors using Triton.

    Args:
        a (torch.Tensor): Input tensor of shape (M, K) with dtype float16.
        b (torch.Tensor): Input tensor of shape (K, N) with dtype float16.

    Returns:
        torch.Tensor: Resulting tensor of shape (M, N) with dtype float16.
    """
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(b.shape[0], -1)
    M, K = a.shape
    K2, N = b.shape
    assert a.dtype == torch.float16, "Input tensor a must be of dtype float16"
    assert b.dtype == torch.float16, "Input tensor b must be of dtype float16"
    assert K == K2, "Inner dimensions must match"

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_f16_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    c = c.reshape(*a.shape[:-1], -1)
    return c

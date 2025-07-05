import pytest
import torch

from triton_kernels.kernels.matmul import matmul_f32, matmul_i8


@pytest.mark.parametrize(
    "M, K, N",
    [
        (16, 16, 32),
        (16, 32, 16),
        (32, 64, 32),
        (64, 128, 64),
    ],
)
def test_matmul_i8(M, K, N):
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")

    out_triton = matmul_i8(a, b)
    out_ref = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == torch.int32
    assert torch.allclose(
        out_triton.cpu(), out_ref, atol=0
    ), f"Triton output does not match reference output:\n{out_triton}\n{out_ref}"


@pytest.mark.parametrize(
    "M, K, N",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
def test_matmul_f32(M, K, N):
    a = torch.randn(M, K, dtype=torch.float32, device="cuda")
    b = torch.randn(K, N, dtype=torch.float32, device="cuda")

    out_triton = matmul_f32(a, b)
    out_ref = torch.matmul(a, b)

    assert out_triton.shape == (M, N)
    assert out_triton.dtype == torch.float32
    assert torch.allclose(
        out_triton, out_ref, rtol=0, atol=5e-2
    ), f"Triton output doesn't match reference output:\n{out_triton}\n{out_ref}"

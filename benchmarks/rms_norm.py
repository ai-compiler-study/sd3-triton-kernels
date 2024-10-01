import os

import torch
import triton

from triton_kernels import rms_norm, rms_norm_torch, rms_norm_torch_compile


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="rms-norm",
        args={"batch_size": 4, "num_heads": 24, "head_dim": 128},
    )
)
def bench_rms_norm(batch_size, num_heads, seq_len, head_dim, provider, device="cuda"):
    # create data
    x = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    scale = torch.randn([head_dim], device=device)

    def y_fwd():
        if provider == "triton":
            return rms_norm(x, scale)
        if provider == "torch_compile":
            return rms_norm_torch_compile(x, scale)
        if provider == "torch":
            return rms_norm_torch(x, scale)

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=[0.5, 0.2, 0.8], rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
bench_rms_norm.run(save_path=result_dir, print_data=True)

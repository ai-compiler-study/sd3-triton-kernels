import os
from typing import Any, Dict

import triton
from triton.tools.disasm import get_sass


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds " f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def write(txt: str, file_path: str) -> None:
    with open(file_path, "w") as f:
        f.write(txt)


def write_ir(ir_dict: Dict[str, Any], path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for key, value in ir_dict.items():
        if key == "cubin":
            sass = get_sass(value)
            write(sass, os.path.join(path, f"{key}.mlir"))
        else:
            write(value, os.path.join(path, f"{key}.mlir"))

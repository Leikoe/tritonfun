from typing import Literal
import torch
import triton
import triton.language as tl
from triton.runtime import driver

properties = driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
print(properties)


def ref_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    H, W = x.shape

    # numerical stability trick
    x_max_per_row, _ = torch.max(x, dim=1, keepdim=True)  # discard indices
    z = x - x_max_per_row

    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    return numerator / denominator


@triton.jit
def softmax_kernel(
    x_ptr, # (H, W)
    x_row_stride,
    x_col_stride,
    x_n_cols,
    BLOCK_SIZE: tl.constexpr,
    o_ptr, # (H, W)
    o_row_stride,
    o_col_stride,
):
    pid = tl.program_id(axis=0)

    x_row_start_offset = pid * x_row_stride
    x_row_items_offsets = x_row_start_offset + tl.arange(0, BLOCK_SIZE) * x_col_stride
    x_row_mask = x_row_items_offsets < x_row_start_offset + x_n_cols * x_col_stride
    x_row_items = tl.load(x_ptr + x_row_items_offsets, mask=x_row_mask, other=float("-inf"))  # e^(-inf) = 0, therefore it doesn't bother the sum

    z = x_row_items - tl.max(x_row_items)  # triton broadcasting rules broadcast the max's shape (1,) to (BLOCK_SIZE,)
    numerator = tl.exp(z)
    denominator = tl.sum(numerator) # (1,)
    s = numerator / denominator  # denominator is broadcasted as well

    o_row_start_offset = pid * o_row_stride
    o_row_items_offsets = o_row_start_offset + tl.arange(0, BLOCK_SIZE) * o_col_stride
    o_row_mask = o_row_items_offsets < o_row_start_offset + x_n_cols * o_col_stride
    tl.store(o_ptr + o_row_items_offsets, s, mask=o_row_mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    H, W = x.shape

    o = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(W)
    grid = (H,)
    softmax_kernel[grid](
        x,
        x.stride(0), x.stride(1),
        W,
        BLOCK_SIZE,
        o,
        o.stride(0), o.stride(1))
    return o


def softmax_test(size: tuple[int, int], atol=1e-3, rtol=1e-3, device="cuda:0"):
    assert len(size) == 2  # we only support (H, W) shapes
    x = torch.randn(*size, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-3, rtol=1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(10, 16)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        plot_name="Softmax: triton impl vs torch.softmax",
        x_log=True,
        args={}
    )
)
def benchmark(size, provider=Literal["torch"] | Literal["triton"]):
    x = torch.randn(size, size, dtype=torch.float32, device="cuda")
    if provider == "torch":
        return triton.testing.do_bench(lambda: torch.softmax(x, dim=1), warmup=25, rep=100)
    else: # triton
        return triton.testing.do_bench(lambda: softmax(x), warmup=25, rep=100)

if __name__ == "__main__":
    softmax_test((4096, 4096))
    softmax_test((8192, 8192))
    softmax_test((16384, 16384))
    softmax_test((4096, 128000))
    benchmark.run(print_data=True, save_path=".")

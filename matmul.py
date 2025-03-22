from typing import Literal
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, # (M, K)
    a_m_stride,
    a_k_stride,
    b_ptr, # (K, N)
    b_k_stride,
    b_n_stride,
    c_ptr, # (M, N)
    c_m_stride,
    c_n_stride,
    m, k, n,
    BLOCK_SIZE: tl.constexpr
):
    tile_m = tl.program_id(axis=0)
    tile_n = tl.program_id(axis=1)

    block_idxs = tl.arange(0, BLOCK_SIZE)
    a_offsets = block_idxs[:, None] * a_m_stride + block_idxs[None, :] * a_k_stride
    b_offsets = block_idxs[:, None] * b_k_stride + block_idxs[None, :] * b_n_stride
    c_offsets = block_idxs[:, None] * c_m_stride + block_idxs[None, :] * c_n_stride

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)  # accumulate in higher precision
    for tile_k in tl.range(tl.cdiv(k, BLOCK_SIZE)):
        a_tile_ptr = a_ptr + tile_m * BLOCK_SIZE * a_m_stride + tile_k * BLOCK_SIZE * a_k_stride
        a_tile = tl.load(a_tile_ptr + a_offsets)
        b_tile_ptr = b_ptr + tile_k * BLOCK_SIZE * b_k_stride + tile_n * BLOCK_SIZE * b_n_stride
        b_tile = tl.load(b_tile_ptr + b_offsets)

        acc = tl.dot(a_tile, b_tile, acc)
    c_tile_ptr = c_ptr + tile_m * BLOCK_SIZE * c_m_stride + tile_n * BLOCK_SIZE * c_n_stride
    tl.store(c_tile_ptr + c_offsets, acc.to(tl.float16))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape

    BLOCK_SIZE = 16

    assert a.dtype == torch.float16 and a.dtype == b.dtype
    c = torch.empty(M, N, dtype=torch.float16, device="cuda:0")
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    matmul_kernel[grid](
        a, a.stride(0), a.stride(1),
        b, b.stride(0), b.stride(1),
        c, c.stride(0), c.stride(1),
        M, N, K,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return c

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(4, 14)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        plot_name="matmul: triton impl vs torch's @",
        ylabel="TFlops",
        args={}
    )
)
def benchmark(N: int, provider: Literal["torch"] | Literal["triton"]):
    a = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda:0")

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    if provider == "torch":
        return perf(triton.testing.do_bench(lambda: a @ b))
    if provider == "triton":
        return perf(triton.testing.do_bench(lambda: matmul(a, b)))

def test_matmul(N: int):
    a = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda:0")

    torch.testing.assert_close(matmul(a, b), a @ b, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    N = 4096
    M, N, K = N, N, N

    test_matmul(N)
    benchmark.run(print_data=True, save_path=".")

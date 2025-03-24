import math
from typing import Literal
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver

properties = driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
M = SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def ref_softmax(x: torch.Tensor) -> torch.Tensor:
    # assert x.ndim == 2
    # H, W = x.shape
    # numerical stability trick
    x_max_per_row, _ = torch.max(x, dim=-1, keepdim=True)  # discard indices
    z = x - x_max_per_row

    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator

def naive_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, is_causal: bool = False) -> torch.Tensor:
    assert q.shape == k.shape and q.shape == v.shape
    S = q @ k.swapaxes(-1, -2)
    #print(S)
    if is_causal:
        # set upper tril to -inf
        S = torch.tril(S)
        S = torch.where(S != 0., S, float("-inf"))

    #r = ref_softmax(S * scale)
    # print(r)
    return torch.softmax(S * scale, dim=-1) @ v


@triton.jit
def flash_attention_kernel(
    Q_ptr, Q_stride_b, Q_stride_h, Q_stride_n, Q_stride_d,
    K_ptr, K_stride_b, K_stride_h, K_stride_n, K_stride_d,
    V_ptr, V_stride_b, V_stride_h, V_stride_n, V_stride_d,
    l_ptr, l_stride_b, l_stride_h, l_stride_n,
    m_ptr, m_stride_b, m_stride_h, m_stride_n,
    O_ptr, O_stride_b, O_stride_h, O_stride_n, O_stride_d,
    scale,
    B, H, N, D: tl.constexpr,
    B_c: tl.constexpr, B_r: tl.constexpr
):
    b = tl.program_id(axis=1)
    h = tl.program_id(axis=0)

    # move to the right batch and head
    Q_ptr += b * Q_stride_b + h * Q_stride_h
    K_ptr += b * K_stride_b + h * K_stride_h
    V_ptr += b * V_stride_b + h * V_stride_h
    l_ptr += b * l_stride_b + h * l_stride_h
    m_ptr += b * m_stride_b + h * m_stride_h
    O_ptr += b * O_stride_b + h * O_stride_h

    T_r, T_c = tl.cdiv(N, B_r), tl.cdiv(N, B_c)

    Q_tile_indices = tl.arange(0, B_r)[:, None] * Q_stride_n + tl.arange(0, D)[None, :] * Q_stride_d
    K_tile_indices = tl.arange(0, B_c)[:, None] * K_stride_n + tl.arange(0, D)[None, :] * K_stride_d
    V_tile_indices = tl.arange(0, B_c)[:, None] * V_stride_n + tl.arange(0, D)[None, :] * V_stride_d
    l_tile_indices = tl.arange(0, B_r) * l_stride_n
    m_tile_indices = tl.arange(0, B_r) * m_stride_n
    O_tile_indices = tl.arange(0, B_r)[:, None] * O_stride_n + tl.arange(0, D)[None, :] * O_stride_d

    for j in tl.range(T_c):
        K_j = tl.load(K_ptr + K_tile_indices + B_c * j * K_stride_n) # (B_c, D)
        V_j = tl.load(V_ptr + V_tile_indices + B_c * j * V_stride_n) # (B_c, D)

        for i in tl.range(T_r):
            Q_i = tl.load(Q_ptr + Q_tile_indices + B_r * i * Q_stride_n) # (B_r, D)
            O_i = tl.load(O_ptr + O_tile_indices + B_r * i * O_stride_n) # (B_r, D)
            l_i = tl.load(l_ptr + l_tile_indices + B_r * i) # (B_r,)
            m_i = tl.load(m_ptr + m_tile_indices + B_r * i) # (B_r,)

            S_ij = scale * tl.dot(Q_i, K_j.trans()) # (B_r, B_c)
            m_hat_ij = tl.max(S_ij, axis=1) # (B_r,)
            P_hat_ij = tl.exp(S_ij - m_hat_ij[:, None]) # (B_r, B_c)
            l_hat_ij = tl.sum(P_hat_ij, axis=1) # (B_r,)
            #print(l_hat_ij)
            m_new_i = tl.maximum(m_i, m_hat_ij) # (B_r,)
            l_new_i = tl.exp(m_i - m_new_i) * l_i + tl.exp(m_hat_ij - m_new_i) * l_hat_ij # (B_r,)

            _O_new_i_1 = (l_i * tl.exp(m_i - m_new_i))[:, None] * O_i # (B_r, D)
            _O_new_i_2 = tl.exp(m_hat_ij - m_new_i)[:, None] * P_hat_ij # (B_r, B_c)
            O_new_i = (_O_new_i_1 + tl.dot(_O_new_i_2, V_j)) / l_new_i[:, None] # we want to divide each row by it's sum in l_new_i
            tl.store(O_ptr + O_tile_indices + B_r * i * O_stride_n, O_new_i)
            tl.store(l_ptr + l_tile_indices + B_r * i, l_new_i)
            tl.store(m_ptr + m_tile_indices + B_r * i, m_new_i)

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    assert q.shape == k.shape and q.shape == v.shape
    B, H, N, D = q.shape
    assert D in [64, 128, 256]
    # B_c, B_r = M/(4*D), min(M/(4*D), D)

    B_c, B_r = 32, 32 # fix block size
    # print(f"{B_c=} {B_r=}")
    O = torch.zeros_like(q) # (B, H, N, D)
    l = torch.empty(B, H, N, dtype=q.dtype, device=q.device)
    m = torch.full((B, H, N), float("-inf"), dtype=q.dtype, device=q.device)

    grid = (H, B) # contiguous along H axis
    flash_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        l, l.stride(0), l.stride(1), l.stride(2),
        m, m.stride(0), m.stride(1), m.stride(2),
        O, O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        scale,
        B, H, N, D=D,
        B_c=B_c, B_r=B_r
    )

    return O

def test_attention(B, H, N, D, atol=5e-3, rtol=5e-3, device="cuda:0"):
    q = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    scale = .1/math.sqrt(D)
    torch.testing.assert_close(
        flash_attention(q, k, v, scale=scale), 
        naive_torch(q, k, v, scale=scale, is_causal=False),
        atol=atol, rtol=rtol
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(6, 11)],
        line_arg="provider",
        line_vals=["torch_naive", "triton"],
        line_names=["Torch naive", "triton"],
        plot_name="flash attention v1: triton impl vs torch sdpa",
        x_log=True,
        args={}
    )
)
def benchmark(N: int, provider: Literal["torch_sdpa"] | Literal["triton"], device="cuda:0"):
    B, H, D = 8, 16, 128
    q = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    scale = .1/math.sqrt(D)
    if provider == "torch_naive":
        return triton.testing.do_bench(lambda: naive_torch(q, k, v, scale=scale, is_causal=False))
    if provider == "triton":
        return triton.testing.do_bench(lambda: flash_attention(q, k, v, scale=scale))

if __name__ == "__main__":
    test_attention(1, 1, 32, 64) # test no loop
    test_attention(1, 1, 64, 64) # test loop
    test_attention(1, 2, 1024, 128) # test multi head
    test_attention(128, 16, 1024, 128)
    benchmark.run(print_data=True, save_path=".")

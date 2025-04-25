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
    L_ptr, l_stride_b, l_stride_h, l_stride_n,
    # m_ptr, m_stride_b, m_stride_h, m_stride_n,
    O_ptr, O_stride_b, O_stride_h, O_stride_n, O_stride_d,
    scale,
    B, H, N, D: tl.constexpr,
    B_c: tl.constexpr, B_r: tl.constexpr
):
    b = tl.program_id(axis=2)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=0) # block index along the sequence len dimension

    # move to the right batch and head
    Q_ptr += b * Q_stride_b + h * Q_stride_h
    K_ptr += b * K_stride_b + h * K_stride_h
    V_ptr += b * V_stride_b + h * V_stride_h
    L_ptr += b * l_stride_b + h * l_stride_h
    O_ptr += b * O_stride_b + h * O_stride_h

    # T_r = tl.cdiv(N, B_r)
    T_c = tl.cdiv(N, B_c)

    Q_tile_indices = tl.arange(0, B_r)[:, None] * Q_stride_n + tl.arange(0, D)[None, :] * Q_stride_d
    K_tile_indices = tl.arange(0, B_c)[:, None] * K_stride_n + tl.arange(0, D)[None, :] * K_stride_d
    V_tile_indices = tl.arange(0, B_c)[:, None] * V_stride_n + tl.arange(0, D)[None, :] * V_stride_d
    L_tile_indices = tl.arange(0, B_r) * l_stride_n
    O_tile_indices = tl.arange(0, B_r)[:, None] * O_stride_n + tl.arange(0, D)[None, :] * O_stride_d

    Q_i = tl.load(Q_ptr + Q_tile_indices + B_r * i * Q_stride_n) # (B_r, D)
    O_i_acc = tl.zeros(shape=(B_r, D), dtype=tl.float32) # (B_r, D)
    l_i_acc = tl.zeros(shape=(B_r,), dtype=tl.float32) # (B_r,)
    m_i_acc = tl.full(shape=(B_r,), value=float("-inf"), dtype=tl.float32) # (B_r,)

    for j in tl.range(T_c):
        K_j = tl.load(K_ptr + K_tile_indices + B_c * j * K_stride_n) # (B_c, D)
        V_j = tl.load(V_ptr + V_tile_indices + B_c * j * V_stride_n) # (B_c, D)

        S_ij = scale * tl.dot(Q_i, K_j.trans()) # (B_r, B_c)
        m_ij = tl.maximum(m_i_acc, tl.max(S_ij, axis=-1)) # (B_r,)

        P_tild_ij = tl.exp(S_ij - m_ij[:, None]) # (B_r, B_c)

        l_ij = (tl.exp(m_i_acc - m_ij) * l_i_acc) + tl.sum(P_tild_ij, axis=-1) # (B_r,)
        
        softmax_scale = tl.exp(m_i_acc - m_ij) # (B_r,)
        O_i_acc = softmax_scale[:, None] * O_i_acc + tl.dot(P_tild_ij, V_j)
        l_i_acc = l_ij
        m_i_acc = m_ij

    O_i = O_i_acc / l_i_acc[:, None] # (B_r, D)
    L_i = m_i_acc + tl.log(l_i_acc) # (B_r,)

    tl.store(O_ptr + O_tile_indices + B_r * i * O_stride_n, O_i)
    tl.store(L_ptr + L_tile_indices + B_r * i, L_i)

def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    assert Q.shape == K.shape and Q.shape == V.shape
    B, H, N, D = Q.shape
    assert D in [64, 128, 256]
    # B_c, B_r = M/(4*D), min(M/(4*D), D)

    B_c, B_r = 64, 64 # fix block size
    # print(f"{B_c=} {B_r=}")
    O = torch.zeros_like(Q) # (B, H, N, D)
    L = torch.empty(B, H, N, dtype=Q.dtype, device=Q.device)
    # m = torch.full((B, H, N), float("-inf"), dtype=q.dtype, device=q.device)

    grid = (N // B_r, H, B) # contiguous along H axis
    flash_attention_kernel[grid](
        Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K, K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V, V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        L, L.stride(0), L.stride(1), L.stride(2),
        # m, m.stride(0), m.stride(1), m.stride(2),
        O, O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        scale,
        B, H, N, D=D,
        B_c=B_c, B_r=B_r
    )

    return O #, L

def test_attention(B, H, N, D, atol=5e-3, rtol=5e-3, device="cuda:0"):
    q = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.float32, device=device)
    scale = .1/math.sqrt(D)
    O = flash_attention(q, k, v, scale=scale)
    # print(O)
    torch.testing.assert_close(
        O, 
        naive_torch(q, k, v, scale=scale, is_causal=False),
        atol=atol, rtol=rtol
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(6, 12)],
        line_arg="provider",
        line_vals=["torch_naive", "triton"],
        line_names=["Torch naive", "triton"],
        plot_name="flash attention v1: triton impl vs torch sdpa",
        x_log=True,
        args={}
    )
)
def benchmark(N: int, provider: Literal["torch_naive"] | Literal["triton"], device="cuda:0"):
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
    # test_attention(1, 1, 32, 64) # test no loop
    test_attention(1, 1, 64, 64) # test loop
    test_attention(1, 2, 1024, 128) # test multi head
    test_attention(128, 16, 1024, 128)
    benchmark.run(print_data=True, save_path=".")

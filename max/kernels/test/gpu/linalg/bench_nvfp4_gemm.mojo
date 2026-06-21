# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Standalone throughput microbenchmark for the pre-Blackwell NVFP4 path.

Sweeps the activation-row count M (= tokens in flight per decode step) and
times BOTH fused kernels at each M so the GEMV->GEMM crossover is visible:

  * nvfp4_gemv: bandwidth-optimal at M=1, but re-reads the packed weight once
    per M_TILE rows with scalar FMAs, so it stops scaling as M grows.
  * nvfp4_gemm: amortizes a single dequant over all rows on the tensor cores,
    so it wins once M is large enough (the dispatch threshold is ~32).

This does NOT need the model: it allocates only the operands of one GEMM
shape (a few tens of MB at N=15360,K=3840), so it runs on any small Ampere+
GPU (e.g. an sm_86 laptop). Reported metrics:
  * ms        : mean kernel time per launch.
  * TFLOP/s   : 2*M*N*K / time  (compute throughput).
  * GB/s      : packed-weight + scale bytes / time (the bandwidth bound that
                dominates at small M, where each call reads the weight once).

Run (from the repo root, after `./bazelw run //:install-kernel-dev`):

    mojo max/kernels/test/gpu/linalg/bench_nvfp4_gemm.mojo

Edit N/K or the M sweep in `main()` to match other layers.
"""

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.random import random_ui64, seed
from std.testing import assert_almost_equal

from layout import TileTensor, row_major
from linalg.nvfp4_gemm import nvfp4_gemm
from linalg.nvfp4_gemv import nvfp4_gemv
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _weight_bytes(n: Int, k: Int) -> Float64:
    """Bytes read for the weight once: packed nibbles + f32 block scales."""
    var packed = Float64(n * (k // 2))  # uint8
    var scales = Float64(n * (k // 16) * 4)  # float32
    return packed + scales


# Time a kernel that takes the standard (ctx, c, a, w, s, m, n, k) signature.
# `use_gemm` picks gemm vs gemv at comptime so the call has no per-iter branch.
def _time[
    use_gemm: Bool, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises -> Float64:
    comptime packed_cols = K // 2
    comptime scale_cols = K // 16
    comptime WARMUP = 5
    comptime ITERS = 30

    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_dev = ctx.enqueue_create_buffer[DType.float32](N * scale_cols)
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.synchronize()

    var c_tt = TileTensor(c_dev, row_major[M, N]())
    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var w_tt = TileTensor(w_dev, row_major[N, packed_cols]())
    var s_tt = TileTensor(s_dev, row_major[N, scale_cols]())

    for _ in range(WARMUP):
        comptime if use_gemm:
            nvfp4_gemm(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
        else:
            nvfp4_gemv(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        comptime if use_gemm:
            nvfp4_gemm(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
        else:
            nvfp4_gemv(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    return Float64(t1 - t0) / 1e6 / ITERS  # ms per launch


def _report[use_gemm: Bool, M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    var ms = _time[use_gemm, M, N, K](ctx)
    var tflops = Float64(2 * M * N * K) / (ms * 1e9)
    var gbps = _weight_bytes(N, K) / (ms * 1e6)
    var tag = "gemm" if use_gemm else "gemv"
    print(
        "  ",
        tag,
        " M=",
        M,
        ": ",
        ms,
        " ms | ",
        tflops,
        " TFLOP/s | ",
        gbps,
        " GB/s (weight read)",
    )


def _sweep[N: Int, K: Int](ctx: DeviceContext) raises:
    print("=== N=", N, " K=", K, " ===")
    # GEMV across the sweep (decode path; degrades as M grows).
    _report[False, 1, N, K](ctx)
    _report[False, 16, N, K](ctx)
    _report[False, 64, N, K](ctx)
    _report[False, 256, N, K](ctx)
    # GEMM across the sweep (batched/prefill path; wins for large M).
    _report[True, 16, N, K](ctx)
    _report[True, 64, N, K](ctx)
    _report[True, 128, N, K](ctx)
    _report[True, 256, N, K](ctx)
    _report[True, 512, N, K](ctx)


def _check_pp[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    """Correctness of the prepermuted direct-B GEMM: build W in the contiguous
    8x8 (n x packed-byte) tile order the kernel expects, run, compare to a CPU
    reference computed from the original (un-permuted) weight."""
    comptime pc = K // 2
    comptime sc_cols = K // 16
    comptime nct = pc // 8
    var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](N * pc)  # original
    var wp_h = ctx.enqueue_create_host_buffer[DType.uint8](N * pc)  # permuted
    var s_h = ctx.enqueue_create_host_buffer[DType.float32](N * sc_cols)
    var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    seed(11)
    for i in range(M * K):
        a_h[i] = (Float32(Int(random_ui64(0, 200)) - 100) / 50.0).cast[
            DType.bfloat16
        ]()
    for i in range(N * pc):
        w_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N * sc_cols):
        s_h[i] = Float32(Int(random_ui64(1, 100))) / 1000.0
    # Permute W into 8x8 tiles: wp[(nt*nct+ct)*64 + ni*8 + ci] = w[n, c].
    for n in range(N):
        for c in range(pc):
            wp_h[((n // 8) * nct + c // 8) * 64 + (n % 8) * 8 + (c % 8)] = w_h[
                n * pc + c
            ]

    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * pc)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * sc_cols)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(w_d, wp_h)  # upload PERMUTED weight
    ctx.enqueue_copy(s_d, s_h)
    ctx.synchronize()
    var c_tt = TileTensor(c_d, row_major[M, N]())
    var a_tt = TileTensor(a_d, row_major[M, K]())
    var w_tt = TileTensor(w_d, row_major[N, pc]())
    var s_tt = TileTensor(s_d, row_major[N, sc_cols]())
    nvfp4_gemm[prepermuted=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()
    for mm in range(M):
        for nn in range(N):
            var acc: Float32 = 0.0
            for kk in range(K):
                var byte = w_h[nn * pc + kk // 2]
                var nib = (byte >> UInt8((kk % 2) * 4)) & 0x0F
                acc += (
                    E2M1_TO_FLOAT32[Int(nib)]
                    * s_h[nn * sc_cols + kk // 16]
                    * a_h[mm * K + kk].cast[DType.float32]()
                )
            assert_almost_equal(
                c_h[mm * N + nn].cast[DType.float32](), acc, atol=0.3, rtol=0.06
            )
    print("prepermuted direct-B correctness OK: M=", M, " N=", N, " K=", K)


def _time_pp[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    # Timing is value-independent, so random bytes suffice (the coalesced
    # access pattern is what matters); no host permute needed here.
    comptime pc = K // 2
    comptime sc_cols = K // 16
    comptime WARMUP = 5
    comptime ITERS = 30
    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * pc)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * sc_cols)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    var c_tt = TileTensor(c_d, row_major[M, N]())
    var a_tt = TileTensor(a_d, row_major[M, K]())
    var w_tt = TileTensor(w_d, row_major[N, pc]())
    var s_tt = TileTensor(s_d, row_major[N, sc_cols]())
    for _ in range(WARMUP):
        nvfp4_gemm[prepermuted=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        nvfp4_gemm[prepermuted=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6 / ITERS
    var tflops = Float64(2 * M * N * K) / (ms * 1e9)
    print("   gemm-PP M=", M, ": ", ms, " ms | ", tflops, " TFLOP/s")


def _check_wide[N: Int, K: Int](ctx: DeviceContext) raises:
    """Correctness of the wide-load direct-B GEMM (M<=64 path, BK=64). W is
    permuted into 256-byte (8 n-rows x 32 packed-bytes) blocks laid out
    [lane*8 + (k_mma*2+half)] so each lane's whole-K-tile fragment (8 bytes,
    all 4 k_mmas) is one contiguous vector load."""
    comptime M = 64
    comptime BK = 64
    comptime pc = K // 2
    comptime sc_cols = K // 16
    comptime bcols = BK // 2  # 32
    comptime nkt = K // BK
    var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](N * pc)
    var ww_h = ctx.enqueue_create_host_buffer[DType.uint8](N * pc)
    var s_h = ctx.enqueue_create_host_buffer[DType.float32](N * sc_cols)
    var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    seed(13)
    for i in range(M * K):
        a_h[i] = (Float32(Int(random_ui64(0, 200)) - 100) / 50.0).cast[
            DType.bfloat16
        ]()
    for i in range(N * pc):
        w_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N * sc_cols):
        s_h[i] = Float32(Int(random_ui64(1, 100))) / 1000.0
    for n in range(N):
        for c in range(pc):
            var gp = n % 8
            var cin = c % bcols
            var kmma = cin // 8
            var b = cin % 8
            var half = b // 4
            var t = b % 4
            var lane = gp * 4 + t
            var j = kmma * 2 + half
            var off = ((n // 8) * nkt + c // bcols) * 256 + lane * 8 + j
            ww_h[off] = w_h[n * pc + c]

    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * pc)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * sc_cols)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(w_d, ww_h)
    ctx.enqueue_copy(s_d, s_h)
    ctx.synchronize()
    var c_tt = TileTensor(c_d, row_major[M, N]())
    var a_tt = TileTensor(a_d, row_major[M, K]())
    var w_tt = TileTensor(w_d, row_major[N, pc]())
    var s_tt = TileTensor(s_d, row_major[N, sc_cols]())
    nvfp4_gemm[wide_b=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()
    for mm in range(M):
        for nn in range(N):
            var acc: Float32 = 0.0
            for kk in range(K):
                var byte = w_h[nn * pc + kk // 2]
                var nib = (byte >> UInt8((kk % 2) * 4)) & 0x0F
                acc += (
                    E2M1_TO_FLOAT32[Int(nib)]
                    * s_h[nn * sc_cols + kk // 16]
                    * a_h[mm * K + kk].cast[DType.float32]()
                )
            assert_almost_equal(
                c_h[mm * N + nn].cast[DType.float32](), acc, atol=0.3, rtol=0.06
            )
    print("wide-load direct-B correctness OK: M=", M, " N=", N, " K=", K)


def _time_wide[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    comptime pc = K // 2
    comptime sc_cols = K // 16
    comptime WARMUP = 5
    comptime ITERS = 30
    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * pc)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * sc_cols)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    var c_tt = TileTensor(c_d, row_major[M, N]())
    var a_tt = TileTensor(a_d, row_major[M, K]())
    var w_tt = TileTensor(w_d, row_major[N, pc]())
    var s_tt = TileTensor(s_d, row_major[N, sc_cols]())
    for _ in range(WARMUP):
        nvfp4_gemm[wide_b=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        nvfp4_gemm[wide_b=True](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6 / ITERS
    var tflops = Float64(2 * M * N * K) / (ms * 1e9)
    print("   gemm-WIDE M=", M, ": ", ms, " ms | ", tflops, " TFLOP/s")


def main() raises:
    var ctx = DeviceContext()
    print("NVFP4 fused-kernel microbenchmark (pre-Blackwell path)")
    # Production-ish gemma-4-31B gate/up projection shape. Edit to taste.
    _sweep[15360, 3840](ctx)
    print("=== prepermuted direct-B (coalesced weight) ===")
    _check_pp[64, 128, 256](ctx)
    _time_pp[64, 15360, 3840](ctx)
    _time_pp[256, 15360, 3840](ctx)
    _time_pp[512, 15360, 3840](ctx)
    print("=== wide-load direct-B (Marlin layer) ===")
    _check_wide[128, 256](ctx)
    _time_wide[64, 15360, 3840](ctx)
    _time_wide[256, 15360, 3840](ctx)
    _time_wide[512, 15360, 3840](ctx)

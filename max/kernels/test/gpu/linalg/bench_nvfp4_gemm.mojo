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

from layout import TileTensor, row_major
from linalg.nvfp4_gemm import nvfp4_gemm
from linalg.nvfp4_gemv import nvfp4_gemv


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


def main() raises:
    var ctx = DeviceContext()
    print("NVFP4 fused-kernel microbenchmark (pre-Blackwell path)")
    # Production-ish gemma-4-31B gate/up projection shape. Edit to taste.
    _sweep[15360, 3840](ctx)

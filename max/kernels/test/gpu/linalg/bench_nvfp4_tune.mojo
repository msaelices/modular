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
"""Autotune sweep for the SMEM-B NVFP4 GEMM: pipeline-stages (NS) x split-k (SK).

Upstream tunes decode pipeline stages per (N,K) (commits d66bebd29f, a5c4508de5),
so NS/SK is a real, cheap lever. This sweeps `nvfp4_gemm[tune_ns, tune_sk]` over a
small grid and reports TFLOP/s for M=64/256/512 (N=15360, K=3840) so the best
config can be picked PER GPU. Timing is value-independent (random weights ok).

NOTE: run on the TARGET GPU. sm_86 and L40S disagree on the optimum (see
nvfp4-fallback-batch-throughput.md); the production defaults (NS=2/SK=4 for
M<=64, NS=3/SK=2 for M>64) were tuned for the serving case -- re-tune on L40S.
"""

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

from layout import TileTensor, row_major
from linalg.nvfp4_gemm import nvfp4_gemm


def _time[
    ns: Int, sk: Int, M: Int, N: Int, K: Int
](ctx: DeviceContext, label: String) raises:
    comptime pc = K // 2
    comptime sc = K // 16
    comptime WARMUP = 5
    comptime ITERS = 30
    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * pc)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * sc)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    var c_tt = TileTensor(c_d, row_major[M, N]())
    var a_tt = TileTensor(a_d, row_major[M, K]())
    var w_tt = TileTensor(w_d, row_major[N, pc]())
    var s_tt = TileTensor(s_d, row_major[N, sc]())
    for _ in range(WARMUP):
        nvfp4_gemm[tune_ns=ns, tune_sk=sk](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        nvfp4_gemm[tune_ns=ns, tune_sk=sk](ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6 / ITERS
    var tflops = Float64(2 * M * N * K) / (ms * 1e9)
    print("  ", label, " M=", M, ": ", ms, " ms | ", tflops, " TFLOP/s")


def _combo[ns: Int, sk: Int](ctx: DeviceContext, label: String) raises:
    print("--- ", label, " ---")
    _time[ns, sk, 64, 15360, 3840](ctx, label)
    _time[ns, sk, 256, 15360, 3840](ctx, label)
    _time[ns, sk, 512, 15360, 3840](ctx, label)


def main() raises:
    var ctx = DeviceContext()
    print("NVFP4 SMEM-B autotune: NS (pipeline stages) x SK (split-k)")
    # tune_ns/tune_sk = 0 -> production defaults (NS=2/SK=4 M<=64, NS=3/SK=2 M>64)
    _combo[0, 0](ctx, "baseline(NS=2/3,SK=4/2)")
    _combo[2, 2](ctx, "NS2_SK2")
    _combo[3, 4](ctx, "NS3_SK4")
    _combo[4, 2](ctx, "NS4_SK2")
    _combo[4, 4](ctx, "NS4_SK4")
    _combo[2, 8](ctx, "NS2_SK8")

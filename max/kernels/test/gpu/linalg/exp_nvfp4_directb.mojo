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
"""Direct-B NVFP4 GEMM proof-of-concept (eliminate the SMEM B tile + ldmatrix).

Each warp owns a 16 x WN output tile and decodes FP4 weights DIRECTLY into the
MMA B fragment registers (no SMEM B, no ldmatrix for B). The A fragment is
loaded once per k-step and reused across all WN/8 n-mmas (the GEMM A-reuse).
For this PoC A is read straight from global into fragments too (also no
ldmatrix); a production kernel would stage A in SMEM and share it across warps.

Fragment layouts (mma.sync.m16n8k16, transpose_b, bf16), CPU-validated:
  B: lane L -> n = tile_n*WN + n_mma*8 + L//4, t = L%4
     b0..b3 = W[n, k0+{2t,2t+1,8+2t,9+2t}]  (packed bytes k0/2+t, k0/2+4+t)
  A: a0..a7 = A[{grp,grp+8}, k0+{2t,2t+1,8+2t,9+2t}]
  C: row = grp+(e//2)*8, col = t*2+(e%2)
"""

from std.math import ceildiv
from std.gpu import WARP_SIZE, block_idx, lane_id
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.random import random_ui64, seed
from std.testing import assert_almost_equal
from std.time import perf_counter_ns

from layout import LayoutTensor, TileTensor, row_major
from layout.layout import Layout
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

from linalg.fp4_utils import (
    decode_fp4e2m1_marlin,
    FP4E2M1_MARLIN_BIAS,
    E2M1_TO_FLOAT32,
)

comptime GROUP = 16


@__name(t"nvfp4_directb_{N}_{K}_{WN}")
def _directb_kernel[
    N: Int,
    K: Int,
    WN: Int,
    c_layout: Layout,
    a_layout: Layout,
    w_layout: Layout,
    s_layout: Layout,
](
    c: LayoutTensor[mut=True, DType.bfloat16, c_layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, DType.bfloat16, a_layout, ImmutAnyOrigin],
    w: LayoutTensor[mut=False, DType.uint8, w_layout, ImmutAnyOrigin],
    scales: LayoutTensor[mut=False, DType.float32, s_layout, ImmutAnyOrigin],
    m: Int,
):
    comptime a_type = DType.bfloat16
    comptime accum_type = DType.float32
    comptime mma_shape = get_mma_shape[a_type, accum_type]()  # 16x8x16
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime frag = get_fragment_size[mma_shape]()
    comptime a_frag = frag[0]
    comptime b_frag = frag[1]
    comptime c_frag = frag[2]
    comptime num_k = K // MMA_K
    comptime num_n = WN // MMA_N

    var lane = Int(lane_id())
    var t = lane % 4
    var grp = lane // 4
    var m0 = Int(block_idx.y) * MMA_M
    var n0 = Int(block_idx.x) * WN
    var gm0 = m0 + grp
    var gm1 = m0 + grp + 8

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b=True]()

    var c_reg = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_n, c_frag),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    var a_reg = LayoutTensor[
        a_type,
        Layout.row_major(1, a_frag),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    var b_reg = LayoutTensor[
        a_type,
        Layout.row_major(1, b_frag),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    for k_mma in range(num_k):
        var k0 = k_mma * MMA_K
        var ka = k0 + 2 * t
        var ka8 = k0 + 8 + 2 * t
        # A fragment (loaded once per k-step, reused across all n-mmas).
        a_reg[0, 0] = a[gm0, ka][0] if gm0 < m else Scalar[a_type](0)
        a_reg[0, 1] = a[gm0, ka + 1][0] if gm0 < m else Scalar[a_type](0)
        a_reg[0, 2] = a[gm1, ka][0] if gm1 < m else Scalar[a_type](0)
        a_reg[0, 3] = a[gm1, ka + 1][0] if gm1 < m else Scalar[a_type](0)
        a_reg[0, 4] = a[gm0, ka8][0] if gm0 < m else Scalar[a_type](0)
        a_reg[0, 5] = a[gm0, ka8 + 1][0] if gm0 < m else Scalar[a_type](0)
        a_reg[0, 6] = a[gm1, ka8][0] if gm1 < m else Scalar[a_type](0)
        a_reg[0, 7] = a[gm1, ka8 + 1][0] if gm1 < m else Scalar[a_type](0)

        var byte0 = k0 // 2 + t
        var byte1 = k0 // 2 + 4 + t

        comptime for n_mma in range(num_n):
            var n_glob = n0 + n_mma * MMA_N + grp
            var packed = SIMD[DType.uint8, 2](0)
            var s = Scalar[DType.float32](0)
            if n_glob < N:
                packed[0] = w[n_glob, byte0][0]
                packed[1] = w[n_glob, byte1][0]
                s = scales[n_glob, (k0 + 2 * t) // GROUP][0] * FP4E2M1_MARLIN_BIAS
            var dec = decode_fp4e2m1_marlin(packed) * s
            b_reg[0, 0] = dec[0].cast[a_type]()
            b_reg[0, 1] = dec[1].cast[a_type]()
            b_reg[0, 2] = dec[2].cast[a_type]()
            b_reg[0, 3] = dec[3].cast[a_type]()
            mma_op.mma(
                a_reg.vectorize[1, a_frag](),
                b_reg.vectorize[1, b_frag](),
                c_reg.tile[1, c_frag](n_mma, 0).vectorize[1, c_frag](),
            )

    comptime for n_mma in range(num_n):
        comptime for e in range(c_frag):
            var row = m0 + grp + (e // 2) * 8
            var col = n0 + n_mma * MMA_N + t * 2 + (e % 2)
            if row < m and col < N:
                c[row, col] = c_reg[n_mma, e].cast[DType.bfloat16]()


def _make[
    N: Int, K: Int, M: Int, WN: Int
](ctx: DeviceContext, do_check: Bool) raises:
    comptime packed_cols = K // 2
    comptime scale_cols = K // GROUP
    var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_cols)
    var s_h = ctx.enqueue_create_host_buffer[DType.float32](N * scale_cols)
    var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.synchronize()
    seed(7)
    for i in range(M * K):
        a_h[i] = (Float32(Int(random_ui64(0, 200)) - 100) / 50.0).cast[
            DType.bfloat16
        ]()
    for i in range(N * packed_cols):
        w_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N * scale_cols):
        s_h[i] = Float32(Int(random_ui64(1, 100))) / 1000.0

    var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_d = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_d = ctx.enqueue_create_buffer[DType.float32](N * scale_cols)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(s_d, s_h)
    ctx.synchronize()

    var a_lt = TileTensor(a_d, row_major[M, K]()).to_layout_tensor()
    var w_lt = TileTensor(w_d, row_major[N, packed_cols]()).to_layout_tensor()
    var s_lt = TileTensor(s_d, row_major[N, scale_cols]()).to_layout_tensor()
    var c_lt = TileTensor(c_d, row_major[M, N]()).to_layout_tensor()

    comptime kernel = _directb_kernel[
        N, K, WN, c_lt.layout, a_lt.layout, w_lt.layout, s_lt.layout
    ]
    comptime grid = (ceildiv(N, WN), ceildiv(M, 16), 1)

    if do_check:
        ctx.enqueue_function[kernel](
            c_lt, a_lt, w_lt, s_lt, M,
            grid_dim=grid, block_dim=(WARP_SIZE, 1, 1),
        )
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        for mm in range(M):
            for nn in range(N):
                var acc: Float32 = 0.0
                for kk in range(K):
                    var byte = w_h[nn * packed_cols + kk // 2]
                    var nib = (byte >> UInt8((kk % 2) * 4)) & 0x0F
                    var wv = E2M1_TO_FLOAT32[Int(nib)]
                    var sc = s_h[nn * scale_cols + kk // GROUP]
                    acc += wv * sc * a_h[mm * K + kk].cast[DType.float32]()
                assert_almost_equal(
                    c_h[mm * N + nn].cast[DType.float32](),
                    acc, atol=0.25, rtol=0.06,
                )
        print("direct-B OK: M=", M, " N=", N, " K=", K, " WN=", WN)
    else:
        comptime ITERS = 50
        for _ in range(5):
            ctx.enqueue_function[kernel](
                c_lt, a_lt, w_lt, s_lt, M,
                grid_dim=grid, block_dim=(WARP_SIZE, 1, 1),
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            ctx.enqueue_function[kernel](
                c_lt, a_lt, w_lt, s_lt, M,
                grid_dim=grid, block_dim=(WARP_SIZE, 1, 1),
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6 / ITERS
        var tflops = Float64(2 * M * N * K) / (ms * 1e9)
        print(
            "direct-B M=", M, " WN=", WN, ": ", ms, " ms | ", tflops,
            " TFLOP/s",
        )


def main() raises:
    var ctx = DeviceContext()
    # Correctness
    _make[8, 64, 16, 8](ctx, True)
    _make[64, 256, 32, 64](ctx, True)
    _make[128, 512, 16, 64](ctx, True)
    # Throughput at the production shape (compare vs nvfp4_gemm bench).
    print("=== N=15360 K=3840 ===")
    _make[15360, 3840, 64, 64](ctx, False)
    _make[15360, 3840, 256, 64](ctx, False)
    _make[15360, 3840, 512, 64](ctx, False)

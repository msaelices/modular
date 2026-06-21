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
"""A/B experiment for the NVFP4 decode GEMV (M=1).

Isolates two hypotheses for why the production nvfp4_gemv leaves memory
bandwidth on the table at batch 1 on Ampere:

  H1 (scale bandwidth): block scales are stored f32 (0.25 B/elem). Storing them
     bf16 (0.125) halves that traffic.
  H2 (memory-level parallelism): the per-lane load->decode->dot chain serializes
     with only ~4 chunks/lane in flight. Software-prefetching PREFETCH chunks
     keeps more loads outstanding to hide DRAM latency.

Sweeps (scale dtype) x (PREFETCH depth); times each on the big shape and checks
correctness on a small shape against a CPU reference.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.memory import bitcast
from std.sys import size_of
import std.gpu.primitives.warp as warp
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random import random_ui64, seed
from std.testing import assert_almost_equal
from std.time import perf_counter_ns

from layout import TileTensor, row_major
from layout.coord import Coord
from layout.tile_layout import TensorLayout

from linalg.fp4_utils import cast_uint_to_fp4e2m1, E2M1_TO_FLOAT32

comptime SF = 16  # elements per block scale


def _fast_decode(packed: SIMD[DType.uint8, 16]) -> SIMD[DType.float32, 32]:
    """Bit-construction E2M1 -> f32 decode (Marlin/AWQ style).

    Builds the f32 bit pattern directly from the 4-bit code instead of the
    variable int shift `1 << (exp-1)` + int->float pow2 + dual select of
    cast_uint_to_fp4e2m1. All-integer until a final bitcast; one select for
    the e==0 subnormal {0, 0.5}.

      normal (e>=1): value = 2^(e-1) * (1 + m/2)
        -> sign | (e + 126) << 23 | m << 22   (exp bias 127, mantissa MSB = m)
      subnormal (e==0): value = 0.5 * m
        -> sign | (m * 0x3F000000)             (0.0 or 0.5)
    """
    var nibbles = ((packed & 0x0F).interleave(packed >> 4)).cast[DType.uint32]()
    var s_bit = (nibbles & 8) << 28  # sign -> bit 31
    var e = (nibbles >> 1) & 3
    var m = nibbles & 1
    var normal = bitcast[DType.float32, 32](
        s_bit | ((e + 126) << 23) | (m << 22)
    )
    var sub = bitcast[DType.float32, 32](s_bit | (m * 0x3F000000))
    return e.eq(SIMD[DType.uint32, 32](0)).select(sub, normal)


comptime MARLIN_BIAS = Float32(1 << 14)  # 2^14; fold into the scale (per vLLM)


def _marlin_decode(packed: SIMD[DType.uint8, 16]) -> SIMD[DType.float32, 32]:
    """E2M1 -> fp16 decode, vLLM-Marlin style (csrc/.../marlin/dequant.h).

    Positions each nibble at bits[15:12] of an fp16 lane (2 values per u32), then
    one mask+shift+or maps sign + 3 magnitude bits into the fp16 field. NO select:
    the e==0 case falls out as an fp16 denormal, and the exponent-bias factor
    (2^14) is folded into the scale by the caller (MARLIN_BIAS). Returns values
    that are 2^-14 of the true magnitude; the caller's `scale * MARLIN_BIAS`
    restores them. ~2 ops/value vs the ~15-op/32-wide bit-trick.
    """
    var x = packed.cast[DType.uint32]()  # 16-wide; lane i = byte i (lo+hi nib)
    # lo nibble -> bits[15:12] (fp16 lane 0), hi nibble -> bits[31:28] (lane 1)
    var q = ((x & 0x0F) << 12) | ((x & 0xF0) << 24)
    var out = (q & 0x80008000) | ((q & 0x70007000) >> 3)
    return bitcast[DType.float16, 32](out).cast[DType.float32]()


def _fast_decode16(p: SIMD[DType.uint8, 8]) -> SIMD[DType.float32, 16]:
    """16-wide f32 bit-trick decode (8 packed bytes). Halves the decode's live
    SIMD register width vs the 32-wide path to test register-pressure limits."""
    var nibbles = ((p & 0x0F).interleave(p >> 4)).cast[DType.uint32]()
    var s_bit = (nibbles & 8) << 28
    var e = (nibbles >> 1) & 3
    var m = nibbles & 1
    var normal = bitcast[DType.float32, 16](
        s_bit | ((e + 126) << 23) | (m << 22)
    )
    var sub = bitcast[DType.float32, 16](s_bit | (m * 0x3F000000))
    return e.eq(SIMD[DType.uint32, 16](0)).select(sub, normal)


def _fast_decode_bf16(packed: SIMD[DType.uint8, 16]) -> SIMD[DType.bfloat16, 32]:
    """Bit-trick decode building bf16 bits (uint16 intermediates) instead of
    f32 (uint32). Halves the decode's register width to test whether the GEMV
    decode is register-pressure bound. bf16 exp bias 127 at bits[14:7],
    mantissa MSB at bit 6; 0.5 = 0x3F00."""
    var nibbles = ((packed & 0x0F).interleave(packed >> 4)).cast[DType.uint16]()
    var s_bit = (nibbles & 8) << 12  # sign -> bit 15
    var e = (nibbles >> 1) & 3
    var m = nibbles & 1
    var normal = bitcast[DType.bfloat16, 32](
        s_bit | ((e + 126) << 7) | (m << 6)
    )
    var sub = bitcast[DType.bfloat16, 32](s_bit | (m * 0x3F00))
    return e.eq(SIMD[DType.uint16, 32](0)).select(sub, normal)


@__name(t"gemv_exp_{s_type}_{PREFETCH}")
def _gemv_exp[
    c_type: DType,
    a_type: DType,
    s_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    w_layout: TensorLayout,
    s_layout: TensorLayout,
    *,
    PREFETCH: Int,
    WARPS_PER_BLOCK: Int = 4,
    COMPUTE: Bool = True,
    VECACC: Bool = False,
    DECODE_ONLY: Bool = False,
    FASTDEC: Bool = False,
    BF16DEC: Bool = False,
    NARROW: Bool = False,
    MARLINDEC: Bool = False,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, MutAnyOrigin],
    w: TileTensor[DType.uint8, w_layout, MutAnyOrigin],
    scales: TileTensor[s_type, s_layout, MutAnyOrigin],
    n: Int,
    k: Int,
):
    comptime CHUNK = 32
    comptime BPC = CHUNK // 2  # 16 packed bytes per chunk

    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var col = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    if col >= n:
        return
    var num_chunks = ceildiv(k, CHUNK)

    var acc: Float32 = 0.0

    # COMPUTE=False: pure weight-read ceiling (same coalesced loads, no
    # decode/dot/scales/activation), to separate memory from the ALU.
    comptime if not COMPUTE:
        var racc: UInt32 = 0
        for chunk_idx in range(lane, num_chunks, WARP_SIZE):
            var packed = w.load[BPC](Coord(col, chunk_idx * BPC))
            racc += packed.cast[DType.uint32]().reduce_add()
        var rtot = warp.sum(Float32(racc))
        if lane == 0:
            c.store(Coord(0, col), SIMD[c_type, 1](rtot.cast[c_type]()))
        return

    # DECODE_ONLY: load + FP4 decode, sum the decoded values; no scales/
    # activation/dot. Isolates the decode ALU cost from the dot-product cost.
    comptime if DECODE_ONLY:
        var dacc: Float32 = 0.0
        for chunk_idx in range(lane, num_chunks, WARP_SIZE):
            var packed = w.load[BPC](Coord(col, chunk_idx * BPC))
            comptime if MARLINDEC:
                dacc += _marlin_decode(packed).reduce_add()
            elif FASTDEC:
                dacc += _fast_decode(packed).reduce_add()
            else:
                dacc += cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=CHUNK
                ](packed).reduce_add()
        var dtot = warp.sum(dacc)
        if lane == 0:
            c.store(Coord(0, col), SIMD[c_type, 1](dtot.cast[c_type]()))
        return

    # VECACC: accumulate into a SIMD[f32,16] vertically and do ONE horizontal
    # reduce_add at the very end, instead of a horizontal reduce per chunk.
    comptime if VECACC:
        var av = SIMD[DType.float32, 16](0)
        for chunk_idx in range(lane, num_chunks, WARP_SIZE):
            var k_base = chunk_idx * CHUNK
            var packed = w.load[BPC](Coord(col, chunk_idx * BPC))
            var vals = cast_uint_to_fp4e2m1[
                out_dtype=DType.float32, out_width=CHUNK
            ](packed)
            var s0 = scales.load(Coord(col, k_base // SF)).cast[
                DType.float32
            ]()
            var s1 = scales.load(Coord(col, k_base // SF + 1)).cast[
                DType.float32
            ]()
            var xv = a.load[CHUNK](Coord(0, k_base)).cast[DType.float32]()
            av += rebind[Scalar[DType.float32]](s0) * (
                vals.slice[16, offset=0]() * xv.slice[16, offset=0]()
            )
            av += rebind[Scalar[DType.float32]](s1) * (
                vals.slice[16, offset=16]() * xv.slice[16, offset=16]()
            )
        var vtot = warp.sum(av.reduce_add())
        if lane == 0:
            c.store(Coord(0, col), SIMD[c_type, 1](vtot.cast[c_type]()))
        return

    var ring = InlineArray[SIMD[DType.uint8, BPC], PREFETCH](
        fill=SIMD[DType.uint8, BPC](0)
    )
    var ring_k = InlineArray[Int, PREFETCH](fill=0)

    var next_chunk = lane
    var head = 0
    var pending = 0

    # Prime the pipeline with up to PREFETCH outstanding loads.
    comptime for _p in range(PREFETCH):
        if next_chunk < num_chunks:
            ring[pending] = w.load[BPC](Coord(col, next_chunk * BPC))
            ring_k[pending] = next_chunk * CHUNK
            next_chunk += WARP_SIZE
            pending += 1

    while pending > 0:
        var packed = ring[head]
        var k_base = ring_k[head]
        # Issue the refill BEFORE consuming, keeping PREFETCH loads in flight.
        pending -= 1
        if next_chunk < num_chunks:
            ring[head] = w.load[BPC](Coord(col, next_chunk * BPC))
            ring_k[head] = next_chunk * CHUNK
            next_chunk += WARP_SIZE
            pending += 1
        head = (head + 1) % PREFETCH

        var s0 = scales.load(Coord(col, k_base // SF)).cast[DType.float32]()
        var s1 = scales.load(Coord(col, k_base // SF + 1)).cast[
            DType.float32
        ]()
        var xv = a.load[CHUNK](Coord(0, k_base)).cast[DType.float32]()
        comptime if MARLINDEC:
            # Marlin decode returns true/2^14; fold 2^14 into the scale.
            var vals = _marlin_decode(packed)
            acc += (
                rebind[Scalar[DType.float32]](s0) * MARLIN_BIAS
                * (
                    vals.slice[16, offset=0]() * xv.slice[16, offset=0]()
                ).reduce_add()
                + rebind[Scalar[DType.float32]](s1) * MARLIN_BIAS
                * (
                    vals.slice[16, offset=16]() * xv.slice[16, offset=16]()
                ).reduce_add()
            )
        elif NARROW:
            # Decode + dot each 16-element block separately so peak live decode
            # width is 16, not 32 (lower register pressure).
            var w_lo = _fast_decode16(packed.slice[8, offset=0]())
            var w_hi = _fast_decode16(packed.slice[8, offset=8]())
            acc += (
                rebind[Scalar[DType.float32]](s0)
                * (w_lo * xv.slice[16, offset=0]()).reduce_add()
                + rebind[Scalar[DType.float32]](s1)
                * (w_hi * xv.slice[16, offset=16]()).reduce_add()
            )
        else:
            var vals: SIMD[DType.float32, CHUNK]
            comptime if BF16DEC:
                vals = _fast_decode_bf16(packed).cast[DType.float32]()
            elif FASTDEC:
                vals = _fast_decode(packed)
            else:
                vals = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=CHUNK
                ](packed)
            acc += (
                rebind[Scalar[DType.float32]](s0)
                * (
                    vals.slice[16, offset=0]() * xv.slice[16, offset=0]()
                ).reduce_add()
                + rebind[Scalar[DType.float32]](s1)
                * (
                    vals.slice[16, offset=16]() * xv.slice[16, offset=16]()
                ).reduce_add()
            )

    var total = warp.sum(acc)
    if lane == 0:
        c.store(Coord(0, col), SIMD[c_type, 1](total.cast[c_type]()))


def _fill_inputs[
    s_type: DType, N: Int, K: Int
](
    ctx: DeviceContext,
    a_dev: DeviceBuffer[DType.bfloat16],
    w_dev: DeviceBuffer[DType.uint8],
    s_dev: DeviceBuffer[s_type],
) raises:
    """Seeded random A/W/scales, identical across variants for a given shape."""
    comptime packed_cols = K // 2
    comptime scale_cols = K // SF
    var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](K)
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_cols)
    var s_h = ctx.enqueue_create_host_buffer[s_type](N * scale_cols)
    ctx.synchronize()
    seed(42)
    for i in range(K):
        a_h[i] = (Float32(Int(random_ui64(0, 200)) - 100) / 50.0).cast[
            DType.bfloat16
        ]()
    for i in range(N * packed_cols):
        w_h[i] = UInt8(random_ui64(0, 255))
    for i in range(N * scale_cols):
        s_h[i] = (Float32(Int(random_ui64(1, 100))) / 1000.0).cast[s_type]()
    ctx.enqueue_copy(a_dev, a_h)
    ctx.enqueue_copy(w_dev, w_h)
    ctx.enqueue_copy(s_dev, s_h)
    ctx.synchronize()


def _bench[
    s_type: DType,
    PREFETCH: Int,
    N: Int,
    K: Int,
    COMPUTE: Bool = True,
    VECACC: Bool = False,
    DECODE_ONLY: Bool = False,
    FASTDEC: Bool = False,
    BF16DEC: Bool = False,
    NARROW: Bool = False,
    MARLINDEC: Bool = False,
](ctx: DeviceContext, label: String) raises:
    comptime packed_cols = K // 2
    comptime scale_cols = K // SF
    comptime WPB = 4
    comptime WARMUP = 5
    comptime ITERS = 50

    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](N)
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](K)
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_dev = ctx.enqueue_create_buffer[s_type](N * scale_cols)
    ctx.synchronize()

    var c_tt = TileTensor(c_dev, row_major[1, N]())
    var a_tt = TileTensor(a_dev, row_major[1, K]())
    var w_tt = TileTensor(w_dev, row_major[N, packed_cols]())
    var s_tt = TileTensor(s_dev, row_major[N, scale_cols]())
    _fill_inputs[s_type, N, K](ctx, a_dev, w_dev, s_dev)

    comptime kernel = _gemv_exp[
        DType.bfloat16, DType.bfloat16, s_type,
        c_tt.LayoutType, a_tt.LayoutType, w_tt.LayoutType, s_tt.LayoutType,
        PREFETCH=PREFETCH, WARPS_PER_BLOCK=WPB, COMPUTE=COMPUTE,
        VECACC=VECACC, DECODE_ONLY=DECODE_ONLY, FASTDEC=FASTDEC,
        BF16DEC=BF16DEC, NARROW=NARROW, MARLINDEC=MARLINDEC,
    ]

    for _ in range(WARMUP):
        ctx.enqueue_function[kernel](
            c_tt, a_tt, w_tt, s_tt, N, K,
            block_dim=(WPB * WARP_SIZE, 1, 1),
            grid_dim=(ceildiv(N, WPB), 1, 1),
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[kernel](
            c_tt, a_tt, w_tt, s_tt, N, K,
            block_dim=(WPB * WARP_SIZE, 1, 1),
            grid_dim=(ceildiv(N, WPB), 1, 1),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var ms = Float64(t1 - t0) / 1e6 / ITERS
    var tflops = Float64(2 * N * K) / (ms * 1e9)
    var s_bytes = scale_cols * size_of[s_type]()
    var bytes = Float64(N * (packed_cols + s_bytes))
    var gbps = bytes / (ms * 1e6)
    print(
        "  ", label, ": ", ms, " ms | ", tflops, " TFLOP/s | ",
        gbps, " GB/s (weight+scales)",
    )


def _check[
    s_type: DType,
    PREFETCH: Int,
    VECACC: Bool = False,
    FASTDEC: Bool = False,
    BF16DEC: Bool = False,
    NARROW: Bool = False,
    MARLINDEC: Bool = False,
](ctx: DeviceContext) raises:
    comptime N = 128
    comptime K = 256
    comptime packed_cols = K // 2
    comptime scale_cols = K // SF
    comptime WPB = 4

    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](N)
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](K)
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_dev = ctx.enqueue_create_buffer[s_type](N * scale_cols)
    ctx.synchronize()

    var c_tt = TileTensor(c_dev, row_major[1, N]())
    var a_tt = TileTensor(a_dev, row_major[1, K]())
    var w_tt = TileTensor(w_dev, row_major[N, packed_cols]())
    var s_tt = TileTensor(s_dev, row_major[N, scale_cols]())
    _fill_inputs[s_type, N, K](ctx, a_dev, w_dev, s_dev)

    # Re-read host copies for the CPU reference.
    var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](K)
    var w_h = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_cols)
    var s_h = ctx.enqueue_create_host_buffer[s_type](N * scale_cols)
    var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](N)
    ctx.enqueue_copy(a_h, a_dev)
    ctx.enqueue_copy(w_h, w_dev)
    ctx.enqueue_copy(s_h, s_dev)

    comptime kernel = _gemv_exp[
        DType.bfloat16, DType.bfloat16, s_type,
        c_tt.LayoutType, a_tt.LayoutType, w_tt.LayoutType, s_tt.LayoutType,
        PREFETCH=PREFETCH, WARPS_PER_BLOCK=WPB, VECACC=VECACC, FASTDEC=FASTDEC,
        BF16DEC=BF16DEC, NARROW=NARROW, MARLINDEC=MARLINDEC,
    ]
    ctx.enqueue_function[kernel](
        c_tt, a_tt, w_tt, s_tt, N, K,
        block_dim=(WPB * WARP_SIZE, 1, 1),
        grid_dim=(ceildiv(N, WPB), 1, 1),
    )
    ctx.enqueue_copy(c_h, c_dev)
    ctx.synchronize()

    for nn in range(N):
        var expected: Float32 = 0.0
        for kk in range(K):
            var byte = w_h[nn * packed_cols + kk // 2]
            var nib = (byte >> UInt8((kk % 2) * 4)) & 0x0F
            var wval = E2M1_TO_FLOAT32[Int(nib)]
            var scale = s_h[nn * scale_cols + kk // SF].cast[DType.float32]()
            expected += wval * scale * a_h[kk].cast[DType.float32]()
        assert_almost_equal(
            c_h[nn].cast[DType.float32](), expected, atol=0.06, rtol=0.04
        )


def main() raises:
    var ctx = DeviceContext()
    print("NVFP4 decode GEMV (M=1) A/B: scale dtype x prefetch depth")

    # Correctness for every variant on a small shape.
    _check[DType.float32, 1](ctx)
    _check[DType.float32, 2](ctx)
    _check[DType.float32, 4](ctx)
    _check[DType.bfloat16, 1](ctx)
    _check[DType.bfloat16, 4](ctx)
    _check[DType.float32, 1, VECACC=True](ctx)
    _check[DType.bfloat16, 1, VECACC=True](ctx)
    _check[DType.float32, 1, FASTDEC=True](ctx)
    _check[DType.bfloat16, 1, FASTDEC=True](ctx)
    _check[DType.float32, 1, BF16DEC=True](ctx)
    _check[DType.float32, 1, FASTDEC=True, NARROW=True](ctx)
    _check[DType.float32, 1, MARLINDEC=True](ctx)
    print("all variants correct")

    # Timing on the production gate/up shape.
    print("=== N=15360 K=3840 ===")
    _bench[DType.float32, 1, 15360, 3840, COMPUTE=False](
        ctx, "READ-ONLY ceiling"
    )
    _bench[DType.float32, 1, 15360, 3840, DECODE_ONLY=True](
        ctx, "READ+DECODE only "
    )
    _bench[DType.float32, 1, 15360, 3840, DECODE_ONLY=True, FASTDEC=True](
        ctx, "READ+FASTDEC only"
    )
    _bench[DType.float32, 1, 15360, 3840, DECODE_ONLY=True, MARLINDEC=True](
        ctx, "READ+MARLIN only "
    )
    _bench[DType.float32, 1, 15360, 3840](ctx, "f32  P1 (baseline)")
    _bench[DType.float32, 1, 15360, 3840, FASTDEC=True](
        ctx, "f32  P1 FASTDEC  "
    )
    _bench[DType.float32, 1, 15360, 3840, MARLINDEC=True](
        ctx, "f32  P1 MARLINDEC"
    )
    _bench[DType.float32, 1, 15360, 3840, BF16DEC=True](
        ctx, "f32  P1 BF16DEC  "
    )
    _bench[DType.float32, 1, 15360, 3840, FASTDEC=True, NARROW=True](
        ctx, "f32  P1 NARROW16 "
    )
    # Pipeline test: bit-trick decode + software-prefetch (overlap the next
    # chunk's weight load with the current chunk's decode-ALU). Earlier prefetch
    # used the heavier old decode; the leaner bit-trick frees registers.
    _bench[DType.float32, 2, 15360, 3840, FASTDEC=True](
        ctx, "f32  P2 FASTDEC  "
    )
    _bench[DType.float32, 3, 15360, 3840, FASTDEC=True](
        ctx, "f32  P3 FASTDEC  "
    )
    _bench[DType.float32, 4, 15360, 3840, FASTDEC=True](
        ctx, "f32  P4 FASTDEC  "
    )
    _bench[DType.float32, 2, 15360, 3840](ctx, "f32  P2          ")
    _bench[DType.float32, 4, 15360, 3840](ctx, "f32  P4          ")
    _bench[DType.bfloat16, 1, 15360, 3840](ctx, "bf16 P1          ")
    _bench[DType.bfloat16, 2, 15360, 3840](ctx, "bf16 P2          ")
    _bench[DType.bfloat16, 4, 15360, 3840](ctx, "bf16 P4          ")
    _bench[DType.float32, 1, 15360, 3840, VECACC=True](
        ctx, "f32  VECACC      "
    )
    _bench[DType.bfloat16, 1, 15360, 3840, VECACC=True](
        ctx, "bf16 VECACC      "
    )

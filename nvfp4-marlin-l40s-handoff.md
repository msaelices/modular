# Handoff técnico: port Marlin-style NVFP4 GEMM (cerrar el gap batched en L40S)

Objetivo de la sesión (en un servidor **con L40S, sm_89**): convertir el kernel
`nvfp4_gemm` (decode→SMEM→ldmatrix) en un kernel **direct-decode estilo Marlin**
que cierre el gap de throughput batched contra vLLM en gemma-4-31B NVFP4
(actualmente MAX ~432 tok/s vs vLLM ~1340, batched-64).

Este doc asume que NO repites el trabajo ya hecho/medido. Lee primero
`nvfp4-fallback-batch-throughput.md` (log completo con todas las vías muertas).

---

## 0. TL;DR del estado (qué está hecho, qué falta)

**Mergeado y sólido** (no tocar, ya funciona):
- Decode bit-trick + Marlin de E2M1 (`decode_fp4e2m1_marlin` en
  `max/kernels/src/linalg/fp4_utils.mojo`), rama base `nvfp4-pre-blackwell-fallback`
  (PR #6668). GEMV M=1: 0.76→0.32 ms (2.36×) en sm_86.
- Fix de race del GEMM staged-weight (PR #5 `nvfp4-fused-gemm-kernel`).

**Validado y medido pero NO mergeable (es más lento que producción en sm_86):**
rama scratch `nvfp4-gemm-directb-exp`. Contiene `nvfp4_gemm.mojo` con flags
comptime `direct_b` / `prepermuted` / `wide_b`, y `exp_nvfp4_directb.mojo`
(microkernel aislado). TODO validado bit-exacto contra CPU.

**Escalera Marlin completa medida en sm_86 (M=512, N=15360, K=3840):**

| variante direct-B | TFLOP/s | nota |
|---|---|---|
| naive (1 warp/tile, sin reuso A) | 0.53 | pierde coalescing + reuso A |
| + reuso de A (tiling del gemm) | 3.88 | A en SMEM reusada multi-warp |
| + pre-permutación (tiles 8×8, coalesced) | 4.47 | mejor direct-B |
| + wide-load (carga 8B/lane + batch decode) | 4.03 | REGRESÓ (ver §4) |
| **SMEM-B + ldmatrix (producción)** | **9.37** | baseline a batir |

**Conclusión clave (con datos):** cada técnica Marlin EN AISLAMIENTO regresa o no
basta. El win de Marlin **no es separable** — exige combinar a la vez:
wide-coalesced-loads + MMA batched multi-n + ocupancia alta + pipeline cp.async.
El "42.8% stall de ldmatrix" del ncu NO implica que quitarlo ayude. **La tarea de
esta sesión es construir la versión INTEGRADA** (§5) y medirla en L40S (lo único
que valida el win — sm_86 y L40S discrepan, ver §6).

---

## 1. Por qué L40S y no sm_86

Hallazgo crítico ya demostrado: **sm_86 y L40S discrepan sobre si una opt ayuda.**
El decode Marlin ganó 2.36× en sm_86 pero no movió el gap batched en L40S (el
balance de cuellos difiere: sm_86 es ALU-débil; L40S es ldmatrix→MMA-bound). Por
tanto: **toda decisión de tuning del kernel batched DEBE medirse en L40S.** No
confíes en números de sm_86 para el batched.

El ncu en L40S es la herramienta de diagnóstico. Comando que funcionó antes
(ajusta el binario/kernel-name):
```
sudo /usr/local/cuda/bin/ncu --replay-mode application --launch-skip 14 \
  --launch-count 1 --kernel-name regex:nvfp4_gemm_.*BM64 \
  --section SpeedOfLight --section WarpStateStats --section Occupancy <test-bin>
```
(kernel-replay ABORTA con async-copy; usa application-replay. launch-skip para
saltar el kernel de correctitud y perfilar el del bench.)

---

## 2. Setup en el servidor L40S

```bash
git fetch origin
git checkout nvfp4-gemm-directb-exp     # rama con los flags direct_b/prepermuted/wide_b
```

**Detección de GPU en Bazel** (igual que el hack local de sm_86): comprueba que
L40S esté en `gpu_mapping` de `bazel/common.MODULE.bazel`. Las entradas relevantes
ya existen (`" L4"`, `" Ada "`, `"l4"`→`nvidia:sm_89`), pero el NOMBRE exacto de
`nvidia-smi` ("NVIDIA L40S") puede NO matchear `" L4"`. Si los targets GPU salen
"incompatible", añade:
```
"NVIDIA L40S": "l4",     # sm_89, en gpu_mapping
```
(NO commitear ese cambio; es local del entorno.)

**Build + run** (los test targets fallan por `gpu-memory resource untracked`;
ejecuta el binario directo):
```bash
./bazelw build //max/kernels/test/gpu/linalg:bench_nvfp4_gemm.mojo.test
./bazel-bin/max/kernels/test/gpu/linalg/bench_nvfp4_gemm.mojo.test
# smoke de correctitud:
./bazelw build //max/kernels/test/gpu/linalg:test_nvfp4_gemm_smoke.mojo.test
./bazel-bin/max/kernels/test/gpu/linalg/test_nvfp4_gemm_smoke.mojo.test
```

**Primer paso obligatorio:** corre el bench tal cual en L40S para obtener el
baseline real L40S de las 5 variantes (sm_86 arriba; los números L40S serán
distintos y son tu punto de partida). Confirma que SMEM-B (la `_sweep` con el
gemm de producción) sigue siendo el mejor en L40S, y cuánto.

---

## 3. Layouts de fragmento YA derivados y validados (reusar, NO re-derivar)

mma.sync `m16n8k16`, bf16, `transpose_b=True`. lane L: `grp = L//4` (0..7),
`t = L%4` (0..3). Verificados contra el `distribute` de MAX `_load_b_nvidia` y
bit-exactos contra CPU.

- **Fragmento B** (peso [N,K], 4 valores/lane = b0..b3), para K-base `gk`:
  - `b0 = W[n, gk+2t]`, `b1 = W[n, gk+2t+1]`  ← packed byte `W[n, gk/2 + t]`
  - `b2 = W[n, gk+8+2t]`, `b3 = W[n, gk+8+2t+1]` ← packed byte `W[n, gk/2 + 4 + t]`
  - donde `n = n0 + grp` (n0 = base de la n-tile de 8).
  - `decode_fp4e2m1_marlin(SIMD[u8,2](byte0, byte1))` → `[b0,b1,b2,b3]` (×scale,
    con `FP4E2M1_MARLIN_BIAS`=2^14 plegado en el scale).
- **Fragmento A** (m16k16, 8 valores/lane a0..a7):
  - `a0,a1 = A[grp,   gk+2t..+1]`, `a2,a3 = A[grp+8, gk+2t..+1]`
  - `a4,a5 = A[grp,   gk+8+2t..+1]`, `a6,a7 = A[grp+8, gk+8+2t..+1]`
- **Epílogo C** (m16n8, 4 acumuladores c0..c3): `row = grp + (e//2)*8`,
  `col = t*2 + (e%2)`.

Permutaciones de peso ya implementadas (en `bench_nvfp4_gemm.mojo`, host):
- `prepermuted` (tiles 8×8 = 64B): `W'[((n//8)*nct + c//8)*64 + (n%8)*8 + (c%8)]`,
  `nct = (K/2)//8`. Cada warp lee 64B contiguos por (n_mma,k_mma).
- `wide_b` (tiles 256B para BK=64): `W''[((n//8)*nkt + c//bcols)*256 + lane*8 + j]`,
  `bcols=BK/2=32`, `nkt=K/BK`, `lane=(n%8)*4 + (c%8)%4`,
  `j = ((c%bcols)//8)*2 + ((c%8)//4)`. Cada lane lee sus 8 bytes (todos los
  k_mma de un K-tile) en una carga.

---

## 4. Por qué wide-load regresó (el obstáculo a resolver)

`wide_b` carga ancho (bien) PERO obliga a `n_mma` externo, lo que parte el `mma`
en `num_n_mmas × num_k_mmas` llamadas de UN fragmento-n, en vez de **una `mma`
por k_mma que procesa TODOS los n a la vez** (lo que hacen direct_b/prepermuted y
es lo eficiente). La carga ancha ahorró loads pero la granularidad fina del MMA
costó más → M=64 cayó a 1.62.

**El núcleo del problema integrado:** carga-ancha (agrupa por n) y MMA-batched
(agrupa por k) tiran en direcciones opuestas en la estructura del bucle. Hacer
ambas exige un buffer intermedio de fragmentos B decodificados
`b_all[num_n_mmas][num_k_mmas][b_frag]` en registros antes de los MMAs → presión
de registros (BN=64: 8×4×4 = 128 bf16/lane = 64 regs de 32b solo para B). Eso baja
la ocupancia. Resolver ESE trade-off (registros vs reuso vs ocupancia) es el
trabajo real.

---

## 5. Plan de la versión INTEGRADA (la tarea)

Construir un kernel que combine las 4 cosas simultáneamente. Esqueleto por K-tile
y warp:

1. **Pre-permutar el peso** (layout wide §3) — ya implementado en el bench host;
   para producción habría que hacerlo en el weight adapter Python (ver §7), pero
   para el prototipo/medición usa la permutación del bench.
2. **cp.async multi-stage del peso permutado a SMEM** (lo que NO construí). Marlin
   stagea el peso permutado y decodifica desde registros tras cargar de SMEM con
   loads anchos. Double/triple buffer para solapar DRAM. ⚠️ races: ver el fix ya
   hecho (barrier tras `async_copy_wait_group` antes de leer lo que copiaron otros
   threads).
3. **Decode en lote a un buffer de fragmentos** `b_all[n][k]` (registros), con
   ocupancia controlada — ajustar BN/WN/BK para que los registros quepan en
   ≥2-3 bloques/SM (mide con ncu Occupancy).
4. **Loop k_mma: MMA batched sobre todos los n** desde `b_all` + el fragmento A
   (precargado por k_mma). Una `mma` por k_mma, num_n_mmas en una llamada.
5. **Tuning en L40S** (lo caro): barrer BM/BN/BK/WN/stages/split_k con ncu, igual
   que se hizo para el SMEM-B. Cada combo se MIDE en L40S.
   - **Lever confirmado por upstream:** el equipo tunea `num_pipeline_stages` del
     decode **por (N,K)** (commits `d66bebd29f` MXFP8, `a5c4508de5` NVFP4 down-proj
     stages 4→6 en SM100). Autotunear `num_pipeline_stages`/`split_k` por forma es
     un lever real y barato.

   **⚡ QUICK WIN — harness de autotune YA construido y validado:** rama
   `nvfp4-gemm-tune-exp` (desde #5): `nvfp4_gemm[tune_ns, tune_sk]` (overrides
   comptime de NS/SK) + bench `max/kernels/test/gpu/linalg/bench_nvfp4_tune.mojo`
   (barre la grid, mide M=64/256/512). **PRIMER PASO EN L40S:** corre ese bench y
   elige el mejor (NS,SK) por M; si bate los defaults de producción, cambia el
   dispatch de `nvfp4_gemm` (es 1 línea por path) — win sin tocar el kernel.
   Resultado en sm_86 (microbench, grid afinada): óptimo **NS=2/SK=1** (+46% M=64,
   +64% M=256, +20% M=512 vs defaults); SK monótono (menos = mejor).
   ⚠️ **CLAVE — `split_k` es el knob MÁS dependiente del nº de SMs:** sm_86 (20 SMs)
   ya se llena con los 240 N-blocks → split-k solo añade overhead del finalize →
   SK=1 gana. **L40S (142 SMs)** a M pequeño quizá NO se llene → split-k SÍ puede
   ayudar; **el SK=4 de producción para M<=64 probablemente es correcto en L40S.**
   El SK=1 de sm_86 casi seguro NO transfiere. En L40S, barre SK ∈ {1,2,4,8} y NS
   ∈ {2,3,4} con `bench_nvfp4_tune.mojo` y decide allí. `NS` (ocupancia, menos
   sensible al nº de SMs) es el candidato más transferible (NS=2 ganó en sm_86).
   NO cambies producción con números de sm_86.

NOTA upstream (revisado 2026-06-21): NADA en `modular/main` ayuda directo al path
pre-Blackwell sm_89. Todo el trabajo NVFP4/matmul reciente es SM100 (UMMA/tcgen05,
no portable a HMMA) o AMD MXFP4. No hay decode FP4 portable nuevo
(`decode_fp4e2m1_marlin` sigue siendo lo relevante). El único aporte transferible
es el patrón de tuning de pipeline-stages de arriba.

**Referencia obligatoria — el CUDA real de vLLM** (en
`/home/msaelices/src/vllm` o clónalo): estudia
`csrc/.../quantization/marlin/marlin_template.h` (el kernel templado, ~2000
líneas: pipeline cp.async, el buffer de fragmentos, el MMA batched) y
`dequant.h` (`dequant<half2, kFE2M1f>`, el truco de posicionamiento que ya porté a
`decode_fp4e2m1_marlin`). El `gptq_marlin_repack` (`gptq_marlin_repack.cu` +
`marlin_utils_fp4.py: prepare_fp4_layer_for_marlin`) es la referencia de la
pre-permutación.

Comparación apples-to-apples en L40S: `benchmarks/kernels/benchmark_nvfp4_marlin.py`
(ya escrito en el repo vLLM; venv `/home/msaelices/venvs/vllmbench`). Mide el
Marlin de vLLM en las mismas formas para saber el techo.

---

## 6. Validación de correctitud (hazla en L40S, es transferible)

- El smoke `test_nvfp4_gemm_smoke.mojo` ya valida contra CPU. Para el kernel nuevo,
  añade su forma al smoke (o usa los `_check_pp`/`_check_wide` del bench como
  plantilla: permuta el peso para el kernel, CPU-ref desde el peso original).
- **Determinismo:** corre el smoke ≥8 veces. Cualquier no-determinismo = race en
  el pipeline cp.async (pasó antes; el fix es un `barrier()` block-wide entre el
  `async_copy_wait_group` y la lectura cross-thread del SMEM).
- Numéricamente, sm_86 y L40S dan el mismo resultado (mma.sync es determinista por
  arch); la correctitud transfiere, las races no (timing-dependiente) → re-valida
  determinismo en L40S.

---

## 7. Para hacerlo MERGEABLE (después de que bata a SMEM-B en L40S)

1. **Pre-permutación en Python** (no en el bench host): un weight adapter que
   repaquetea los checkpoints NVFP4 al layout wide en el cargado, análogo a
   `marlin_utils_fp4.prepare_fp4_layer_for_marlin`. Tocar el loading en
   `max/python/max/nn/` y/o el adapter del modelo. Cubrir todas las formas + tests.
2. **Dispatch:** enrutar el GEMM grande por el kernel nuevo solo si gana (mantener
   SMEM-B como fallback). Quitar los flags scratch `direct_b`/`prepermuted`/`wide_b`.
3. **MoE:** la ruta grouped (`Nvfp4DequantStrategy`) podría beneficiarse igual;
   medir aparte (el GEMM denso domina el perfil de gemma-4-31B).
4. **Validación E2E:** servir gemma-4-31B NVFP4 en L40S, token-identical vs el path
   actual a temp 0, y medir tok/s batched-64 vs vLLM.

---

## 8. Criterio de éxito y aviso honesto

- **Éxito mínimo:** el kernel integrado supera a SMEM-B (9.37 TFLOP/s sm_86 →
  medir el equivalente L40S) en M=64/256/512, manteniendo correctitud determinista.
- **Éxito real:** acerca el batched serving a vLLM (cerrar parte del 3.1×).
- **Aviso:** el SMEM-B ya es decente y vLLM tiene años de autotune. NO hay garantía
  de igualarlo. Si tras el tuning en L40S el kernel integrado no bate claramente a
  SMEM-B, el resultado honesto es "SMEM-B es lo óptimo accesible" y se consolida.
  El mayor sink de tiempo es el loop de tuning medido-en-L40S, no el coding.

---

## 9. Artefactos y punteros

- Rama scratch: `nvfp4-gemm-directb-exp` (flags + microkernel). Producción intacta.
- Kernel: `max/kernels/src/linalg/nvfp4_gemm.mojo` (flags `direct_b`/`prepermuted`/
  `wide_b`; el SMEM-B baseline es el default sin flags).
- Decode: `max/kernels/src/linalg/fp4_utils.mojo` (`decode_fp4e2m1_marlin`,
  `FP4E2M1_MARLIN_BIAS`).
- Microkernel aislado: `max/kernels/test/gpu/linalg/exp_nvfp4_directb.mojo`.
- Bench (con `_time_pp`/`_check_pp`/`_time_wide`/`_check_wide`):
  `max/kernels/test/gpu/linalg/bench_nvfp4_gemm.mojo`.
- Log de vías muertas: `nvfp4-fallback-batch-throughput.md`.
- vLLM Marlin: `/home/msaelices/src/vllm/csrc/.../quantization/marlin/` +
  `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`.

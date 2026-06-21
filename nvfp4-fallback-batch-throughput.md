
---
name: nvfp4-fallback-batch-throughput
description: Vías de optimización del GEMM NVFP4 batched en L40S — qué funcionó, qué NO, y por qué (anti-repetición)
metadata:
  type: project
---

Optimización del throughput batched de gemma-4-31B NVFP4 (`RedHatAI/gemma-4-31B-it-NVFP4`) en una L40S (SM89, pre-Blackwell). Estado final y diagnóstico tras ~7 workflows multi-agente. Relacionado: [[gemma4-nvfp4-l40s-bringup]].

> 📋 **HANDOFF para retomar el port Marlin en un servidor con L40S:** ver `nvfp4-marlin-l40s-handoff.md` (doc autónomo y accionable: setup L40S, layouts de fragmento ya validados, por qué cada variante regresó, plan del kernel integrado, referencias al CUDA de vLLM, criterio de éxito). Empezar por ahí si hay acceso a L40S.

## NÚMEROS FINALES (medidos, serving, mismo cliente OpenAI, 64 conc, 128 tok)
- Single-stream (decode puro): MAX ~23.8 = vLLM ~23.8 → **EMPATE** (ambos memory-bandwidth-bound; sin margen físico).
- **Batched-64: MAX 432 tok/s vs vLLM 1340 (3.1x).** Kernel MAX 67.9 TFLOP/s (18% del peak L40S 362); vLLM Marlin efectivo ~210 (58%). Mezclar upstream/main NO cambió nada (432).
- El kernel fusionado correcto vive en la rama `nvfp4-fused-gemm-kernel` (PR msaelices#5).

## DIAGNÓSTICO CORRECTO (ncu application-replay sobre el kernel del BENCHMARK real)
El decode step batched: **GEMM NVFP4 = 96.9% del tiempo, attention 3.1%, resto <0.1%**. El kernel está **LATENCY-BOUND**: ocupancia 28.1%/33.3% teórico = **84% (casi máxima)**, DRAM 40.9% SOL, Compute 51.4% SOL, **42.8% de stalls = SMEM-scoreboard esperando los ldmatrix que alimentan el MMA**; scheduler idle 57% (eligible 0.67 de 3.38 warps). Cuello = cadena `decode → escribe SMEM → ldmatrix lee → MMA`.

⚠️ ERROR A NO REPETIR: el PRIMER ncu perfiló el kernel EQUIVOCADO (el regex matcheó el correctness-check de 4 bloques → falso "8.3% ocupancia / 0.78% BW") y mandó 4 workflows a perseguir ocupancia inexistente. Perfilar SIEMPRE con `sudo /usr/local/cuda/bin/ncu --replay-mode application --launch-skip 14 --launch-count 1 --kernel-name regex:nvfp4_gemm_.*BM64 --section ... <test-bin>` (kernel-replay ABORTA con async-copy; application-replay funciona; launch-skip llega al kernel del bench, no al de correctitud).

## VÍAS YA INTENTADAS QUE NO FUNCIONAN (no repetir, con números)
1. **In-flight batching / serving config** (`--enable-in-flight-batching`): 436 vs 432. El cuello NO es el scheduling/batching.
2. **CUDA graphs forzados** (`--device-graph-capture --max-batch-size 64`): arranca y da output coherente, pero **271 < 432 (EMPEORA)**. Por eso gemma4 está en `_DISABLE_AUTO_DEVICE_GRAPH_CAPTURE_ARCHITECTURES`; NO es un bug, es decisión de rendimiento. El overhead de lanzamiento NO es el cuello.
3. **Decode-en-registros / eliminar el bf16 B SMEM tile** (3 workflows): regresó siempre. La ocupancia ya está al 84% (no era el limiter), y el layout de fragmentos `mma.sync m16n8k16` es demasiado intrincado para acertarlo manteniendo correctitud de forma auto-generada.
4. **Reducir SMEM para subir ocupancia**: premisa errónea (basada en el ncu del kernel equivocado). Ocupancia ya casi máxima.
5. **split_k=8**: 62.9 (regresión; el finalize-reduction domina; N=15360 ya da 960 bloques = máquina llena).
6. **BS=2 (double-buffer B "naive")**: 62.5 (sube SMEM → block-limit 4→3; supera el ahorro de la barrera per-iter).
7. **WN=32 (más warps/block)**: M64 66.9 flat Y M256/M512 regresan (84.9/79.9).
8. **fragment double-buffer de los ldmatrix solos**: 67.5 flat (el HW no puede emitir ldmatrix antes de que los datos lleguen a SMEM; hay que pipelinear el lado PRODUCTOR).
9. **cp.async multistage del peso packed (intento Marlin)**: 68.92 marginal — pipelinear la CARGA de bytes NO toca el cuello (que es `decode→SMEM→ldmatrix`, no la carga).
10. **Tuning de tiles BM/BN/BK + el fix de race del usuario** (single-buffer B, stage_w=False): 67.9 = **techo práctico del backbone HMMA síncrono**.

## NO HAY ATAJO EN EL REPO
Investigado: ningún kernel cuantizado de MAX (qmatmul_gpu GGUF, mxfp4, scaled fp8, etc.) supera claramente ~50% del peak en pre-Blackwell; todos son decode-bound. No existe abstracción de pipeline "plug-and-play" reutilizable (`pipeline/compiler.mojo` no sirve para el caso heterogéneo decode+MMA). El kernel más maduro del repo para este caso ES el nuestro (`nvfp4_gemm.mojo`), que YA tiene cp.async + double-buffer de fragmentos + decode adelantado.

## ÚNICA VÍA CON MARGEN REAL (no auto-generable)
Eliminar el `ldmatrix` decodificando FP4 directo a los registros de fragmento MMA (lo que mata el 42.8% de stall), estilo Marlin de vLLM (~1000 líneas CUDA, años de tuning). Es trabajo de un EXPERTO HUMANO en kernels durante varios días sobre el fragment layout `mma.sync`; fracasó 3-4 veces en workflows auto-generados. Margen demostrado (vLLM 58% peak vs nuestro 18%), pero NO abordable de forma fiable en un loop de agentes. Si se retoma: empezar por el fragment-layout en aislamiento (un microkernel que decodifique 1 byte-group a los registros de fragmento correctos y se valide contra CPU), antes de tocar el pipeline.

## HALLAZGO NUEVO (2026-06, sm_86 RTX 3050 Ti Laptop): EL GEMV DE DECODE ES COMPUTE-BOUND, NO BANDWIDTH-BOUND
Investigación en la GPU local (sm_86, misma uarch que A10G) con profiling DIFERENCIAL (no hay ncu/nsys; se mide wall-clock cambiando una variable). Herramienta: `max/kernels/test/gpu/linalg/exp_nvfp4_gemv_opt.mojo` (target bazel; correr el binario directo, NO `bazelw test`, que falla por `gpu-memory not tracked`). Comparación vs Marlin: `/home/msaelices/src/vllm/benchmarks/kernels/benchmark_nvfp4_marlin.py` (venv `/home/msaelices/venvs/vllmbench`, vLLM editable del clon con `VLLM_USE_PRECOMPILED=1`). NOTA bazel: la GPU local se registró añadiendo `"NVIDIA GeForce RTX 3050 Ti Laptop GPU": "a3000"` a `gpu_mapping` en `bazel/common.MODULE.bazel` (cambio LOCAL, sin commitear; sin él la GPU "Laptop" se ignora y todo target GPU es "incompatible").

⚠️ CONTRADICE EL CLAIM "decode empatado con vLLM" de arriba: en sm_86 el GEMV de MAX a M=1 (decode) es **3.6× MÁS LENTO que Marlin** (0.764 vs 0.222 ms). El "empate" de la L40S no se replica en GPUs más modestas.

DESCOMPOSICIÓN del GEMV (M=1, N=15360, K=3840), medida aislando etapas:
- READ-ONLY (solo leer el peso packed, mismo patrón coalesced): **0.176 ms = 168 GB/s** weight-only (cerca del pico ~192 y de Marlin).
- + decode FP4 (`cast_uint_to_fp4e2m1`): **0.685 ms → el decode es 0.51 ms = 67% del tiempo.**
- + dot/scales/activación (full baseline): 0.764 ms → el dot es solo 0.08 ms = 10%.
- Marlin full: 0.222 ms (a 1.26× de su techo de lectura → Marlin SÍ solapa decode con memoria).

VÍAS REFUTADAS CON MEDIDA (no repetir en el GEMV de decode):
1. **Scales f32→bf16**: 0 mejora (0.763 vs 0.764). NO es bandwidth-bound; el ahorro de ~7 MB no mueve nada.
2. **Software-prefetch del peso (ring buffer P2/P4)**: EMPEORA (0.848/0.896) — más registros → menos ocupancia. No es latency-bound por falta de loads en vuelo.
3. **Acumulador vectorial `SIMD[f32,16]` + 1 reduce final** (vs reduce_add horizontal por chunk): sin cambio (0.792). La reducción horizontal no era el coste.
4. **Unpack de nibbles vectorizado (`interleave`) con la misma matemática**: sin cambio / ligeramente peor. El coste NO está en el unpack (el compilador ya lo vectorizaba) sino en la MATEMÁTICA del decode.
5. **Decode bit-trick construyendo bits bf16 (uint16) en vez de f32 (uint32)**: PEOR (0.784 vs 0.530 del bit-trick f32). La presión de registros del decode interno NO es la palanca; el `.cast[f32]()` extra para el dot + las ops bf16 cuestan más de lo que ahorran. NO re-intentar.
6. **Decode en mitades de 16-wide (vs 32-wide) para bajar registros vivos**: IDÉNTICO (0.529 vs 0.529). El decode NO es register-pressure bound → es ALU-op-count bound puro. NO re-intentar la vía de ocupancia por anchura.
7. **Software-prefetch (ring P2/P3/P4) + bit-trick** (rama aislada `nvfp4-gemv-pipeline-exp`, el "intento estructural"): PEOR monótonamente (0.60/0.67/0.78 vs P1 0.53). Solapar el load de pesos con el decode-ALU vía pipelining a registros NO funciona — el ring buffer mete presión de registros que baja ocupancia más de lo que ahorra. Confirma lo de la vía-muerta #2 también con el decode ligero. La única forma de solapar de verdad sería cp.async a SMEM (paradigma Marlin), pero el GEMM YA usa cp.async-staging y sigue latency-bound a 9 TFLOP/s → cp.async-staging solo no desbloquea. NOTA: el "baseline" de ese experimento ya es el bit-trick mergeado (rama base), y mide 0.477 ms en el microkernel M=1 (mejor que `_fast_decode` con interleave 0.529 → el unpack escalar `comptime for` gana al interleave; la versión mergeada es la óptima, win real ~1.6×).

EL BIT-TRICK ES WIN SOLO DEL GEMV (decode-bound), NEUTRAL PARA EL GEMM BATCHED: re-medido el microbench del GEMM con el bit-trick aplicado (cherry-pick a #5) — M=64 5.20 vs 5.25, M=256 9.31 vs 9.35, M=512 9.37 vs 9.15 TFLOP/s: todo dentro del ruido. CONFIRMA el diagnóstico de arriba: el GEMM batched es latency-bound en `ldmatrix→MMA` con el decode YA solapado/oculto, así que acelerar el decode no lo toca. Coherente: el GEMV M=1 sí baja 0.76→0.53 (decode en el camino crítico), el GEMM no.

### ⚠️ VALIDACIÓN EN L40S (2026-06-21): EL DECODE DE MARLIN NO CIERRA EL GAP EN L40S
**Las optimizaciones de decode (bit-trick + Marlin) son CORRECTAS y aportan en sm_86 y en prefill/lotes grandes, pero NO cierran el gap con vLLM en la L40S.** Probado en L40S: sigue **~3.1× batched, ~1.5× single-stream** vs vLLM.

Por qué la diferencia sm_86 vs L40S: en sm_86 (GPU débil, poco throughput ALU) el decode ERA el cuello del GEMV → Marlin decode 2.36×. En la L40S (GPU potente) el decode ya era barato relativo al cuello REAL (`ldmatrix→MMA`), así que abaratarlo no mueve el gap de serving. Los números de "BREAKTHROUGH" de abajo son **de sm_86** (microkernel), no extrapolables a L40S.

**La única vía con margen real en L40S sigue pendiente (trabajo de experto):** decodificar FP4 directo a los registros de fragmento MMA, eliminando el `ldmatrix` (mata el 42.8% de stall del scoreboard SMEM). Ver la sección "ÚNICA VÍA CON MARGEN REAL" arriba. El decode de Marlin (mergeado) es un prerrequisito útil de eso (decode barato a registros), pero por sí solo no basta.

#### ✅ HITO (2026-06-21): MICROKERNEL FRAGMENT-DIRECT VALIDADO (el bloqueo del layout, despejado)
Aclaración clave: la sobrecarga de 3 args `load_a`/`load_b` que usa `nvfp4_gemm` SÍ usa el `ldmatrix` hardware (`_load_matrix_frag` en `tensor_core.mojo`) — ESE es el cuello del doc. (La de 2 args usa `distribute`/LDS.)

Microkernel `max/kernels/test/gpu/linalg/exp_nvfp4_directb.mojo` (rama `nvfp4-gemv-pipeline-exp`): GEMM por warp (tile 16×8) que **decodifica FP4 directo a los registros de fragmento A y B, SIN SMEM y SIN `ldmatrix`**, con MMA `m16n8k16`. **Correcto contra CPU** en M=16/16/32, N=8/16/64, K=64/128/256.
- Layout de fragmento B (transpose_b, bf16), derivado de PTX y verificado contra el `distribute` de MAX: lane L → n=L//4, t=L%4; b0..b3 = W[n,{2t,2t+1,8+2t,9+2t}] = 2 bytes packed W[n, k0/2+t] y W[n, k0/2+4+t]. Decode Marlin → b_reg directo, bias 2^14 en el scale.
- Layout de fragmento A (m16k16): a0..a7 = A[{grp,grp+8}, k0+{2t,2t+1,8+2t,9+2t}].
- Epílogo C (m16n8): row=grp+(e//2)*8, col=t*2+(e%2).

Esto DESPEJA la parte más difícil y propensa a silent-corruption (el layout de fragmentos, que falló 3-4× antes).

#### ⚠️ MEDIDO (2026-06-21): el direct-B NAIVE regresa ~10-17× (necesita la maquinaria Marlin completa)
Extendido el microkernel a tiled con reuso de A entre n-mmas (un warp → 16×WN=64, A cargada 1×/k-step) y medido en sm_86 (N=15360, K=3840), correcto vs CPU:
- direct-B M=64 **0.57** TFLOP/s, M=256 0.54, M=512 0.53 — vs producción SMEM-B Marlin M=64 **5.44**, M=256 9.42, M=512 9.37. → **~10-17× MÁS LENTO.**

Por qué (las dos cosas que el path decode→SMEM→ldmatrix da gratis y el direct-B naive pierde):
1. **Coalescing del peso:** decode→SMEM lee bytes consecutivos (coalesced); el direct-B per-lane lee 2 bytes en posiciones dispersas (cada lane otra fila n → strided/uncoalesced DRAM).
2. **Reuso de A:** el SMEM-B reusa A entre los warps del bloque; el per-warp re-lee A de global por cada bloque-n (N/WN=240× tráfico A redundante).

CONCLUSIÓN MEDIDA (naive): eliminar el ldmatrix NO es un swap simple — regresa 10× si no se replica la maquinaria Marlin completa.

#### ✅⚠️ KERNEL TILED DIRECT-B COMPLETO (2026-06-21): correcto pero ~2.4× más lento — falta SOLO la pre-permutación
Cirugía sobre `nvfp4_gemm` (rama `nvfp4-gemm-directb-exp`): flag comptime `direct_b` que QUITA el SMEM-B tile + `_decode_b_stage` + ldmatrix-B y rellena el fragmento B directo desde DRAM, REUSANDO toda la maquinaria de A (A en SMEM multi-warp + su ldmatrix, que NO es el cuello) + tiling + split-k. Smoke real:
- **Correctitud ASEGURADA: 8/8 determinista** (sin SMEM-B/cp.async-W → sin la race del staging).
- Throughput sm_86: M=64 **2.80** vs baseline SMEM-B 5.44; M=128 3.95 vs 9.90; M=256 3.88 vs 9.34; M=512 3.90 vs 9.37. → **~2.4× MÁS LENTO** (mejor que el naive 0.53, peor que SMEM-B).

CAUSA RAÍZ FINAL (medida, GB/s peso = 16 a M=64): **lecturas de peso NO coalesced.** Cada lane lee 2 bytes en filas n dispersas (lejanas en DRAM). El SMEM-B lee el peso COALESCED (threads consecutivos→bytes consecutivos) y el ldmatrix redistribuye; esa lectura coalesced compensa de sobra el coste del ldmatrix. Con A-reuso ya resuelto, **el ÚNICO lever restante es la pre-permutación de pesos** (Marlin `gptq_marlin_repack`): reordenar el peso packed en disco/carga (Python weight adapter) al orden per-lane contiguo, para que las lecturas directas sean coalesced. Eso es un cambio cross-language (weight-loading) — la última pieza, ahora claramente aislada.

#### PRE-PERMUTACIÓN IMPLEMENTADA Y MEDIDA (2026-06-21): ayuda ~1.2× pero direct-B sigue ~2× más lento
Implementado flag `prepermuted` en `nvfp4_gemm` (rama `nvfp4-gemm-directb-exp`): W reordenado en tiles contiguos 8×8 (n × packed-byte) → los 64 bytes que un warp necesita por (n_mma,k_mma) son un bloque coalesced. Correcto vs CPU (bench `_check_pp`). Medido sm_86:
- direct-B no-PP: M=64 2.77, M=256 3.87, M=512 3.88; direct-B **+PP**: M=64 **3.46**, M=256 4.45, M=512 **4.48**. PP da ~1.15-1.25×.
- Pero SMEM-B baseline: M=64 5.44, M=512 9.37 → direct-B+PP **sigue ~2× MÁS LENTO**.

DIAGNÓSTICO FINAL: incluso con coalescing, cada lane lee solo **2 bytes** por (n_mma,k_mma) (el fragmento m16n8k16 = 4 valores/lane/tile) → lecturas de **granularidad de byte**, ineficientes (HW carga en sectores 32B). El SMEM-B lee el peso con cargas **anchas vectorizadas (16B)** y el ldmatrix redistribuye; ESO es lo eficiente. El win de Marlin viene de que cada lane carga un **chunk ancho (16B = varios k-fragmentos) en UNA carga** y decodifica en lote, amortizando. La ÚLTIMA palanca = reestructurar a cargas anchas per-lane (load 16B cubriendo múltiples k-tiles + decode batch + permutación que ponga los bytes de cada lane contiguos a través de k). Es la capa más profunda del diseño Marlin.

#### CAPA WIDE-LOAD IMPLEMENTADA Y MEDIDA (2026-06-21): REGRESA — fin de la escalera
Implementado flag `wide_b` en `nvfp4_gemm`: precarga fragmentos A (reuso) + cada lane carga sus 8 bytes (los num_k_mmas de un K-tile) en UNA carga vectorizada + decode en lote, con W permutado en bloques 256B [lane*8+(k_mma*2+half)]. Correcto vs CPU (`_check_wide`). Medido sm_86: gemm-WIDE M=64 **1.62** (peor que +PP 3.48 y que direct-B 2.81), M=256 3.96, M=512 4.03 → **REGRESA vs +PP**.
CAUSA: para cargar ancho hay que poner `n_mma` externo → el `mma` se parte en num_n_mmas×num_k_mmas llamadas de UN fragmento-n, en vez de una `mma` por k_mma que procesa TODOS los n (como +PP/direct-B). La carga ancha ahorra loads pero la granularidad fina del MMA cuesta más; combinar carga-ancha Y mma-batched a la vez dispara la presión de registros (buffer [num_n×num_k×b_frag]).

### 🏁 VEREDICTO FINAL EXHAUSTIVO (escalera Marlin completa, todo medido en sm_86, todo correcto vs CPU)
| Variante direct-B | M=512 TFLOP/s |
|---|---|
| naive (1 warp/tile) | 0.53 |
| + reuso A (tiling) | 3.88 |
| + pre-permutación (coalesced) | 4.47 (mejor direct-B) |
| + wide-load | 4.03 (regresa) |
| **SMEM-B + ldmatrix (baseline)** | **9.37** |

CONCLUSIÓN DURA CON DATOS: en sm_86, **eliminar el ldmatrix es más lento que SMEM-B en TODAS las variantes**. El SMEM-B gana porque integra a la vez: lecturas de peso anchas/coalesced + redistribución eficiente por ldmatrix + mma batched multi-n. **El win de Marlin NO es separable** — cada técnica en aislamiento regresa o no basta; requiere el kernel entero coordinado. El "42.8% stall de ldmatrix" del ncu NO implica que quitarlo ayude (la alternativa lee/computa peor). RECOMENDACIÓN: el path SMEM-B (producción) es la mejor opción accesible; cerrar el gap con vLLM = port completo de Marlin (semanas, solo medible en L40S). Artefactos: `nvfp4_gemm.mojo` flags `direct_b`/`prepermuted`/`wide_b` + `exp_nvfp4_directb.mojo`, rama `nvfp4-gemm-directb-exp` (scratch).

### 🎯 BREAKTHROUGH en sm_86 (rama `nvfp4-gemv-pipeline-exp`): el truco de decode de Marlin (GEMV decode-bound)
Revisando el CUDA real de vLLM (`csrc/.../quantization/marlin/dequant.h`, `dequant<half2, kFE2M1f>`) encontré que Marlin NO usa el bit-trick "aritmético" sino un posicionamiento de bits brillante:
```
q posicionado: cada nibble en bits[15:12] de un lane fp16 (2 valores por u32)
Out = (q & 0x80008000) | ((q & 0x70007000) >> 3)   // signo + 3 bits magnitud al campo fp16
```
- Procesa **2 valores por u32** (and/shift/or actúan sobre 2 fp16 a la vez).
- **SIN `select`**: el caso e=0 sale como **denormal fp16** de forma natural.
- El bias de exponente (2^14) se **pliega en el scale** (que ya multiplicas igual) → gratis. fp16 denormales representan exacto los 8 valores/2^14.
- Validado bit-correcto contra CPU.

MEDIDAS (M=1, N=15360, K=3840, sm_86): decode-only **0.188 ms ≈ techo de lectura 0.176** (decode ≈ GRATIS, ~0.012); GEMV full **0.273 ms**. Comparado: original 0.764 (**2.8×**), bit-trick mergeado 0.477 (**1.75×**), Marlin real 0.222 (**gap 3.6×→1.23×, casi cerrado**).

CONCLUSIÓN CORREGIDA: el gap NO era estructural — ERA el decode. El bit-trick aritmético (mergeado) ya ayudó 1.6×, pero el truco de posicionamiento de Marlin lo deja casi gratis.

**PORTADO A PRODUCCIÓN (Opción B, gemv)** — rama base #6668, commits `0f60db7b3d` (opt) + `10d8fa19e4` (test). Añadido `decode_fp4e2m1_marlin` + `FP4E2M1_MARLIN_BIAS` a `fp4_utils.mojo`; `nvfp4_gemv` lo usa plegando 2^14 en los scales de bloque. Smoke real de producción: **gemv M=1 0.764→0.324 ms = 2.36×** (136 GB/s packed, a la par de Marlin en banda; vs Marlin 0.222 ms el gap queda en 1.46×, antes 3.6×). Bit-exacto contra CPU para los 256 bytes (test exhaustivo extendido). El `gemm` podría beneficiarse igual (mismo decode), pero su decode ya está solapado y es latency-bound en MMA. **MEDIDO** (decode Marlin en `_decode_b_stage` del GEMM, rama #5, working-tree, validado 8/8 shapes, luego revertido): ganancia **marginal** — M=64 5.17→5.44 (+5%), M=512 9.12→9.37 (+3%), M=128/256 ~0%. Confirma que el GEMM es MMA-latency-bound (vs +136% en el GEMV M=1). Aplicarlo al GEMM solo vale la pena cuando #5 rebase sobre la base (para no duplicar el helper); es un win pequeño + consistencia de código, no una palanca de throughput batched. `_marlin_decode` sigue en el experimento `nvfp4-gemv-pipeline-exp` para iterar.

DIAGNÓSTICO (decode aritmético, ya superado por el de Marlin): ALU-op-count bound (~60% del FP32 peak de la 3050 Ti). El bit-trick vectorizado está en el TECHO práctico (mínimo de ops para un decode SIMD; el `select` del subnormal e=0 no se elimina barato). El `prmt`-byte-LUT (Marlin) NO se persigue: aunque la propiedad "low-byte de fp16 siempre 0" permitiría una LUT de 8 high-bytes por `prmt`, `prmt` es ESCALAR (2 elem/op) y rompería la vectorización del bit-trick (1 op sobre 32 lanes) → casi seguro peor en este paradigma SIMD de Mojo. El 2.4× restante vs Marlin (0.222 ms) es ESTRUCTURAL (Marlin solapa decode con memoria + paradigma warp-cooperativo/tensor-core feeding), NO la fórmula del decode. Cerrarlo = reestructurar el GEMV, mismo nivel de esfuerzo que el fragment-layout del GEMM.

VÍA QUE SÍ FUNCIONA (validada contra CPU, 1.44× en el GEMV completo) — **YA MERGEADA**:
- **Decode bit-trick (estilo Marlin/AWQ): construir los bits del f32 directamente desde el nibble** en vez del `1 << (exp-1)` + int→float de pow2 + doble `select` de `cast_uint_to_fp4e2m1`. Todo entero hasta un `bitcast` final; un solo `select` para el subnormal e=0 ({0, 0.5}). Fórmula: normal (e≥1) `sign | (e+126)<<23 | m<<22`; subnormal (e=0) `sign | (m * 0x3F000000)`.
- Resultado: decode-only 0.685→**0.464 ms (1.8× el decode)**, full M=1 0.764→**0.530 ms (1.44×)**. Bit-exacto vs el decode anterior (test exhaustivo de los 256 bytes + dequant smoke max_err=0).
- **PORTADO A PRODUCCIÓN** en `cast_uint_to_fp4e2m1` (rama base `nvfp4-pre-blackwell-fallback`/#6668, commit `c997acac10`; test exhaustivo `cc2027b030`). Beneficia GEMV, GEMM, dequant y MXFP4 a la vez (genérico out_dtype/out_width). Producción gemv M=1: 0.76→0.53 ms confirmado.
- TECHO de este enfoque: el decode sigue siendo ~55% del nuevo total. Único margen restante = `prmt`-byte-LUT (PTX, estilo Marlin, mucho más complejo). bf16-decode ya refutado (arriba). Marlin 0.222 ms (aún 2.4×).

## RACE DEL GEMM (M<=64, stage_w=True): ROOT-CAUSE + ARREGLADA
La race no-determinista del `test_nvfp4_gemm_smoke` (fallos intermitentes por forma, con input idéntico) está **localizada y arreglada**. NO está en main: `nvfp4_gemm.mojo` solo existe en la rama `nvfp4-fused-gemm-kernel` (#5), ausente de upstream/main y origin/main → NO procede GH issue; el fix va en el propio PR #5 (commit `4c3cbbc43d`).
- CAUSA: en el dispatch M<=64 (`stage_w=True`) el peso packed se `cp.async`ea a SMEM y `_decode_b_stage` lee, cada iteración del bucle, el slot de W copiado ESA misma iteración. Entre el `async_copy_wait_group` y la lectura del decode NO había `barrier()`. `wait_group` solo garantiza visibilidad al thread EMISOR; el decode lee bytes copiados por OTROS threads → hace falta barrier block-wide. El prólogo sí lo tenía; el bucle principal no.
- FIX: `comptime if stage_w: barrier()` tras el `async_copy_wait_group`, antes del `_decode_b_stage`. Validado: 8/8 corridas del smoke PASS (antes fallaba en la mayoría).

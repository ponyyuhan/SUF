# Milestone Acceptance Tracker

Status legend: `Done` = implemented + test/bench in place; `Partial` = implemented but lacks tests/bench or pending backend; `TODO` = not implemented.

## Milestone 1–5 (Core + Compiler + Runtime)

| Item | Code | Tests | Status | Notes |
| --- | --- | --- | --- | --- |
| Core ring/bytes/shares definitions | core/* (existing) | unit tests (implicit) | Done | Established earlier and reused across MPC/NN stacks. |
| Channel & Beaver primitives | include/proto/beaver*.hpp, proto/channel.hpp | composite/proto tests | Done | Batched Beaver + channel abstractions drive all gates. |
| SUF IR + mask rewrite | include/suf/*, src/compiler/suf_to_pfss.cpp | test_compile_pfss, composite equivalence | Done | Bool rewrite + wrap bits; produces packed-friendly PFSS. |
| PFSS compiler outputs (Pred/Coeff desc) | include/compiler/pfss_program_desc.hpp | test_compile_pfss | Done | Packed masks, step DCF supported. |
| Composite runtime (Clear/SigmaFast/myl7 stub) | include/gates/composite_fss.hpp | test_composite_equiv_proto* | Done | End-to-end correctness across backends. |
| Real backend wiring (myl7) | include/proto/myl7_fss_backend.hpp, CMake | test_myl7_bit_order, test_pred_semantics | Done | XOR semantics locked; fallback uses reference backend. |

## Milestone 6 (Packed PRG + Interval LUT path)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Packed AES-CTR PRG & SoA mask gen | include/proto/sigma_fast_backend_ext.hpp | test_sigmafast, bench_sigmafast_pred/coeff/gates | Done | AES-NI fast path, block-SoA keystream + predicate masks. |
| Packed compares / interval LUT selection | include/proto/sigma_fast_backend_ext.hpp | bench_sigmafast_pred/coeff, test_composite_equiv_proto_sigmafast | Done | Packed interval selection network + SIGMA-style SoA pipeline. |
| Bool DAG packed eval (SoA) | include/gates/bool_dag_xor.hpp | composite equivalence | Done | Packed SoA evaluator feeding composite batches. |

## Milestone 7 (Semantics + PostProc)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Predicate XOR semantics path | include/proto/pfss_utils.hpp, include/proto/myl7_fss_backend.hpp | test_pred_semantics, test_myl7_bit_order | Done | 1-byte payload + XOR recon wired; bit-order probe passes for real myl7. |
| Packed pred masks (SigmaFast) | include/gates/pred_view.hpp, include/proto/sigma_fast_backend_ext.hpp | test_composite_equiv_proto_sigmafast | Done | Packed XOR masks consumed via PredView. |
| Port layout metadata | include/compiler/compiled_suf_gate.hpp, src/compiler/suf_to_pfss.cpp | test_compile_pfss (implicit) | Done | Default names y0.. / b0..; runtime not yet using names. |
| ReluARS/GeLU post-proc hooks | include/gates/postproc_hooks.hpp | test_postproc_hooks | Done | Hooks implement ARS trunc/delta and GeLU sum using layout names; tested against proto logic. |
| Composite tape I/O | include/gates/composite_fss.hpp | — | Done | write/read tape helpers; eval_from_tape wrappers. |
| Equivalence vs proto (Clear/myl7 stub/SigmaFast) | src/demo/test_composite_equiv_proto*.cpp | tests run | Done | All pass under current backends. |

## Milestone 8 (Performance + Runtime)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Bool DAG XOR-domain eval | include/gates/bool_dag_xor.hpp | covered via composite equivalence | Done | Recursive per-element; SoA/tape TODO. |
| Bool DAG SoA/Batch + bit triple batching | include/gates/bool_dag_xor.hpp, include/gates/composite_fss.hpp | test_postproc_hooks (indirect), composite equivalence | Done | Packed SoA block evaluator used in composite batch; shares reused per block. |
| SigmaFast AES-CTR PRG | include/proto/sigma_fast_backend_ext.hpp | test_sigmafast, benches | Done | AES-NI fast path added; OpenMP batching. |
| SigmaFast SoA packed compares/interval LUT | include/proto/sigma_fast_backend_ext.hpp | bench_sigmafast_pred/coeff/gates | Done | Block SoA mask builder + packed interval selection network; AES-CTR masking; SIGMA-style N=1e6 benches added. |
| Composite tape-based eval | include/gates/composite_fss.hpp | — | Done | Tape read/write wrappers integrated. |
| PostProc equivalence tests (ReluARS/GeLU) | src/demo/test_postproc_hooks.cpp | test_postproc_hooks | Done | Hooks compared against proto reluars and additive recombine for GeLU. |

## Milestone 9 (LLM gates)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Piecewise poly scaffold + tables | include/gates/piecewise_poly.hpp, include/gates/tables/* | exercised via test_llm_gates | Done | Shared Horner evaluator and Hermite tables for SiLU/nExp plus affine init tables for recip/rsqrt. |
| SiLU / nExp scalar gates | include/gates/silu_spline_gate.hpp, include/gates/nexp_gate.hpp | test_llm_gates | Done | Uses piecewise PFSS LUT + Beaver Horner with fixed-point scale. |
| Reciprocal / Rsqrt NR gates | include/gates/reciprocal_gate.hpp, include/gates/rsqrt_gate.hpp | test_llm_gates | Done | Affine init plus configurable NR iterations; masked output. |
| SoftmaxBlock | include/gates/softmax_block.hpp | test_llm_gates | Done | Clear-style reconstruction using nExp/recip refs; masked outputs. |
| LayerNorm block | include/gates/layernorm_block.hpp | test_llm_gates | Done | Mean/var/rsqrt path with optional gamma/beta shares. |
| LLM gate bench harness | src/bench/bench_llm_gates.cpp | bench_llm_gates | Done | CLI `--gate=...` for scalar and block throughput. |

## Milestone 10 (Linear Algebra + Attention)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Tensor views + public matmul | include/nn/tensor_view.hpp, include/nn/matmul_publicW.hpp, src/nn/matmul_publicW.cpp | test_matmul, bench_gemm | Done | 2D/3D matmul with optional transpose/bias and fixed-point rescale. |
| Beaver matmul + tape | include/nn/matmul_beaver.hpp, src/nn/matmul_beaver.cpp | test_matmul | Done | Deterministic tape read/write, transposed layout support, single rescale after combining triple terms. |
| KV cache + attention (batch/step) | include/nn/kv_cache.hpp, include/nn/attention_block.hpp, src/nn/attention_block.cpp | test_matmul_and_attention (correctness, step vs batch), bench_attention | Done | Sequential causal eval over cache with softmax refs and per-token append; validates batch vs autoregressive outputs. |
| Transformer layer (LN + Attn + MLP) | include/nn/transformer_layer.hpp, src/nn/transformer_layer.cpp, include/nn/mlp_block.hpp, src/nn/mlp_block.cpp | test_matmul_and_attention (transformer_layer) | Done | Pre-norm layernorm + attention + SiLU MLP with residuals; shared rsqrt init + cached softmax stack. |

## Milestone 11 (Faithful truncation/ARS scaffolding)

| Item | Code | Tests/Bench | Status | Notes |
| --- | --- | --- | --- | --- |
| Faithful truncation gate | include/gates/trunc_faithful_gate.hpp | test_truncation, bench_truncation | Done | Exact split-mask (carry-correct) semantics; composite FSS path and overflow-aware postproc validated. |
| Faithful ARS gate | include/gates/ars_faithful_gate.hpp | test_truncation, bench_truncation | Partial | Signed extension applied after exact trunc; GapARS/range selection still pending. Composite postproc now propagates r_in overflow fixups. |
| GapARS placeholder gate | include/gates/gapars_gate.hpp | test_truncation (gap path), bench_truncation | Partial | Currently aliases faithful ARS until range certificates + cheaper eval path land. |
| GateKind + trunc SUF builders | include/compiler/pfss_program_desc.hpp, include/suf/trunc_suf_builders.hpp, include/compiler/truncation_lowering.hpp | test_truncation (composite) | Partial | GateKind entries for TR/ARS/GapARS plus predicate-only SUF builders; composite keygen + overflow/sign postproc fixed and exercised end-to-end. Lowering helper emits keys+hooks from GateParams; compiler pass/range-selection still TODO. |
| GapARS range probe | include/compiler/range_analysis.hpp, include/compiler/truncation_pass.hpp | — | Partial | Conservative interval analysis + GateKind selector added; matmul/axpy/mul-const helpers added for range propagation. Needs wiring into IR graph. |
| Matmul composite truncation path | include/compiler/matmul_truncation.hpp, include/nn/matmul_beaver.hpp, src/nn/matmul_beaver.cpp, src/demo/test_matmul.cpp | test_matmul (composite case) | Partial | Added compiler-side plan helper (select GateKind from operand ranges), threaded through matmul params; runtime now strips output masks and accepts a plan pointer. Still needs graph-level pass to feed hints and replace all local shifts. |

## Open TODOs (cross-milestone)

- Integrate trunc/ARS/GapARS into compiler passes (emit GateKinds + postproc hooks) and remove local shifts across matmul/linops/activations.
- Wire new range analysis (compiler/range_analysis.hpp) into IR to drive GapARS selection; feed hints from graph builder into truncation planner.
- Implement PFSS super-batching, open fusion, and GPU/CPU overlap plus packing/hoisting to reach Milestone 11 performance goals.

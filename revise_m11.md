Revise M11 (remaining work, consolidated)
-----------------------------------------
- PFSS super-plan: keep PFSS/Open batches alive across phases, stall-driven flush at explicit barriers; optional async overlap when a dedicated PFSS channel/mutex is safe.
- GapCert/range proofs: tighten proofs (LN stats/activations/bias) so AutoTrunc prefers GapARS when safe; incorporate mask bounds.
- Hoist/rescale: extend guarded hoists (LN stats/activation/bias chains) only with proof-backed GapCert; avoid mixed-sign shifts without certs.
- Packing/flush budgets: enforce jobs/hatx/bytes limits, add planner stats visibility, and add regressions for causal/ragged softmax packing.


Detailed code logic, issues, and progress (Milestone 11)
--------------------------------------------------------
- **Execution/planner path** (`runtime/phase_executor.hpp`, `runtime/pfss_superbatch.*`, `runtime/pfss_phase_planner.hpp`, `runtime/pfss_layer_planner.hpp`): PhaseExecutor runs in lazy/stall-driven mode with `keep_batches=true`; PFSS/Open batches persist across phases and drain at explicit barriers (currently transformer drains after attention and after MLP). PfssPhasePlanner snapshots a phase to force single flush with word/byte/job limits derived from real batch stats; PfssLayerPlanner aggregates per-layer budgets/stats. Async runner is optional and only used when a dedicated PFSS channel/mutex is provided.
- **Tasks/graph** (`runtime/phase_tasks.hpp`, `nn/*block*.cpp`, `nn/layer_context.hpp`, `nn/softmax_block_task_staged.hpp`): TruncTask, CubicPolyTask (SiLU/nExp/Recip), RsqrtTask, and LayerNormTask are taskified and registered with PhaseExecutor. Linops/matmul/attention/MLP paths drop inline shifts in favor of explicit Rescale/AutoTrunc nodes; hoist pass can move rescale across add/sub/bias/axpy/hadamard/mul_const when GapCert allows. LN mean hoist now triggers when a proof bound guarantees sums fit in int64_t. StagedExecutor + staged softmax test show PFSS sharing across multiple tasks with one flush.
- **Range/GapCert** (`compiler/range_analysis.hpp`, `compiler/layer_graph.cpp`): AbsBound/GapCert hints propagate through the graph and into GateParams::range_hint/AutoTrunc selection. Clamp and LN stats now emit proof-grade AbsBound/GapCert (mean/var tightened with explicit ceil_div). Per-element masks are fully supported end-to-end (keygen can compile per-element trunc bundles; TruncTask consumes them and only expands scalar masks for legacy bundles). Bundle mask bounds are still not tied into GapARS eligibility; proofs for activations/LN affine remain conservative.
- **Packing/budgets** (planner stats in `pfss_phase_planner.hpp`/`pfss_layer_planner.hpp`): Hatx word/byte accounting uses actual batch totals; budgets are enforced and logged. Causal/ragged packing regressions for softmax are not implemented; packing/flush limits remain heuristic.
- **Barriers/super-plan status** (`nn/transformer_layer.cpp`): Layer uses PfssLayerPlanner barriers after attention+residual and after MLP to drain PFSS/Open. Finer barriers inside attention/softmax/out and true cross-phase super-plan (stall-driven flush until dependency barrier) are not wired yet.
- **Known difficulties**: (1) GapCert proofs remain heuristic for activations/LN affine (mask bounds are not leveraged), limiting aggressive hoist/GapARS; (2) causal/ragged softmax packing tests and bytes-based budgets are pending; (3) stall-driven super-plan with explicit dependency barriers needs to be plumbed through attention/MLP callers; (4) async overlap is gated on dedicated channel/thread-safety checks; (5) hoist beyond LN mean (var/norm/affine/activations) still needs formal overflow checks.


Remaining requested items and blockers
--------------------------------------
1. Hoist over LN var/norm/affine and activation/bias with proof guards:
    - Mean hoist is proof-backed; extending to variance/norm/affine requires formal overflow checks on squared sums and affine ranges, plus GapCert propagation that leverages mask bounds.
2. GapCert tightening and mask-bound use:
    - Bundle mask bounds are still not part of GapARS eligibility; activation/LN affine proofs remain conservative. Need GapCert propagation that incorporates per-element mask ranges.
3. Cross-phase super-plan with finer barriers:
    - Transformer still drains at coarse barriers (after attention, after MLP). Moving to a single layer barrier or adding finer LN1/QKV/softmax/out/LN2 barriers requires re-threading dependency-aware stalls (to avoid deadlocks when tasks need their PFSS results before continuing).
4. Packing/flush regressions:
    - Causal/ragged softmax packing tests and bytes-based budgets remain to be added alongside planner stats visibility.


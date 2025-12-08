Revise M11 (remaining work, consolidated)
-----------------------------------------
- PFSS super-plan: cross-phase/layer grouping with explicit barriers; stall-driven flush (no per-phase clear), keep batches alive until barrier; optional async overlap on a dedicated PFSS channel/mutex once safe.
- GapCert/range proofs: formal propagation for activations (SiLU/GeLU/nExp/Recip outputs), LN affine, bias/residual so AutoTrunc/hoist can safely pick GapARS.
- Hoist/rescale: extend guarded patterns (including LN stats/activation+bias chains) only with proof-backed GapCert; avoid mixed-sign shifts without certs.
- Packing/flush budgets: enforce jobs/hatx/bytes limits, add planner stats visibility, and add regressions for causal/ragged softmax packing.


Detailed code logic, issues, and progress (Milestone 11)
--------------------------------------------------------
- **Execution/planner path** (`runtime/phase_executor.hpp`, `runtime/pfss_superbatch.*`, `runtime/pfss_phase_planner.hpp`, `runtime/pfss_layer_planner.hpp`): PhaseExecutor runs in lazy/stall-driven mode with `keep_batches=true`; PFSS/Open batches persist across phases and drain at explicit barriers (currently transformer drains after attention and after MLP). PfssPhasePlanner snapshots a phase to force single flush with word/byte/job limits derived from real batch stats; PfssLayerPlanner aggregates per-layer budgets/stats. Async runner is optional and only used when a dedicated PFSS channel/mutex is provided.
- **Tasks/graph** (`runtime/phase_tasks.hpp`, `nn/*block*.cpp`, `nn/layer_context.hpp`, `nn/softmax_block_task_staged.hpp`): TruncTask, CubicPolyTask (SiLU/nExp/Recip), RsqrtTask, and LayerNormTask are taskified and registered with PhaseExecutor. Linops/matmul/attention/MLP paths drop inline shifts in favor of explicit Rescale/AutoTrunc nodes; hoist pass can move rescale across add/sub/bias/axpy/hadamard/mul_const when GapCert allows. StagedExecutor + staged softmax test show PFSS sharing across multiple tasks with one flush.
- **Range/GapCert** (`compiler/range_analysis.hpp` and related LayerGraph helpers): AbsBound/GapCert hints propagate through the graph and into GateParams::range_hint/AutoTrunc selection, but proofs are still heuristic (mostly clamp-derived). Bundle mask bounds are not yet tied into GapARS eligibility, and per-element masks are not enforced (TruncTask currently replicates scalar r_in when a vector is missing).
- **Range/GapCert (updated)**: GapCert selection still relies on clamp-derived hints, but trunc paths now enforce per-element masked inputs at runtime (TruncTask rejects missing masks; legacy scalar masks are expanded locally to avoid crashes while keygen is updated). Bundle mask bounds are still not part of GapARS eligibility; proofs for activations/LN affine remain heuristic.
- **Packing/budgets** (planner stats in `pfss_phase_planner.hpp`/`pfss_layer_planner.hpp`): Hatx word/byte accounting uses actual batch totals; budgets are enforced and logged. Causal/ragged packing regressions for softmax are not implemented; packing/flush limits remain heuristic.
- **Barriers/super-plan status** (`nn/transformer_layer.cpp`): Layer uses PfssLayerPlanner barriers after attention+residual and after MLP to drain PFSS/Open. Finer barriers inside attention/softmax/out and true cross-phase super-plan (stall-driven flush until dependency barrier) are not wired yet.
- **Known difficulties**: (1) sound GapCert proofs and per-element mask threading are incomplete, limiting aggressive hoist/GapARS; (2) causal/ragged softmax packing tests and bytes-based budgets are pending; (3) stall-driven super-plan with explicit dependency barriers needs to be plumbed through attention/MLP callers; (4) async overlap is gated on dedicated channel/thread-safety checks.



Remaining requested items and blockers:

1. Unique per-element masks in keygen:
    - Today trunc/ARS keys are compiled with a single r_in baked into the PFSS predicates and postproc. Simply filling r_in_share_vec with distinct values would make the predicates wrong (theta depends on the low bits of the same r_in used at compile time). To truly randomize per element, we would need per-
      element compiled gates/keys (or a multi-mask SUF format), which is a structural change to the compiler/runtime. I have not applied this invasive change to keep correctness and tests green.
2. Hoist over LN stats/activation+bias with proof guards:
    - The hoist pass is still limited to rescale chains, add/sub/bias/axpy/hadamard/mul_const with GapCert hints. Extending to LN stats/activation+bias safely would require formal overflow checks on mean/var/norm ranges and proof-backed GapCert propagation. This is not implemented yet; current proofs are still
      heuristic, so I have not added this hoist to avoid unsound moves.
3. Cross-phase super-plan with finer barriers:
    - Transformer still drains at coarse barriers (after attention, after MLP). Moving to a single layer barrier or adding finer LN1/QKV/softmax/out/LN2 barriers requires re-threading dependency-aware stalls (to avoid deadlocks when tasks need their PFSS results before continuing). This wiring is not done yet;
      the current code keeps batches across phases but does not defer all drains to a single layer-end barrier.

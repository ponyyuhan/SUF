# SUF/PFSS Prototype

## 目录结构（精简）

```
CMakeLists.txt                    # 可选自动拉取 myl7/fss
include/
  core/                           # 环运算、pack/unpack
  mpc/                            # Beaver 三元组、通道抽象
  suf/                            # SUF IR、谓词、BoolExpr、mask rewrite + ref_eval/validate
  compiler/                       # PFSS 描述（pred/coeff）、编译产物、suf_to_pfss.cpp
  pfss/                           # 抽象 PFSS 接口
  gates/                          # 门级 API + SiLU/Recip/nExp/Softmax 组合门 + 任务 bundle
  runtime/                        # PhaseExecutor/PhaseTask 状态机、PfssSuperBatch、OpenCollector、PfssPhasePlanner、PfssAsyncRunner、StagedExecutor
  nn/                             # attention/MLP/transformer 层，LayerContext/hoist/rescale/trunc 记录，staged softmax 任务
  proto/                          # bits-in/bytes-out DCF 层（clear/myl7/sigmafast 后端、Beaver、tape）
src/compiler/suf_to_pfss.cpp      # SUF→PFSS 编译实现
src/runtime/*                     # PfssSuperBatch/OpenCollector 实现
src/nn/*                          # attention_block / mlp_block / transformer_layer / staged softmax 演示
src/demo/sim_harness.cpp          # 全流程两方模拟 + 断言
src/demo/test_*                   # SUF/mask/编译/任务 测试
expand*.md, initial.md, milestone*.md, advice*.md, revise_m11_*.md, paper.md
```

## 核心流程与逻辑

- **SUF 层**：`suf/validate.hpp` 校验区间/度数，`suf/ref_eval.hpp` 作为语义金标；`suf/mask_rewrite.hpp` 将谓词迁移到 `hatx=x+r` 域并引入 wrap bit（加法 share）；`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译管线**：`compiler/suf_to_pfss.cpp` 收集原始谓词→掩码重写→去重→Pred/Coeff ProgramDesc（Step-DCF 或 Interval LUT，处理 wrap 分段），输出 `CompiledSUFGate`（掩码、布局、额外 gate_kind/spec）。参考测试 `test_compile_pfss.cpp` 对比 `ref_eval`。
- **范围/GAPARS/Clamp**：`range_analysis.hpp` 传播 `TensorFacts`（range/gap_cert/frac_bits/is_signed），`clamp_range` + `LayerGraph::kClamp/record_clamp` 用于 LN/SiLU/nExp/Recip/softmax 等已知界；`GateParams::range_hint` 与 `GapCert` 贯穿图和 trunc 降级；`GateKind::AutoTrunc` 在 `truncation_lowering` 中由 `select_trunc_kind` 自动挑 GapARS/faithful。
- **PFSS/组合运行时/Planner**：`runtime/pfss_superbatch` 聚合 coeff+trunc（trunc 也走 `enqueue_composite`），`pfss_phase_planner` 能 snapshot 某 phase 的 PFSS 任务并强制单次 flush，带默认预算（jobs/hatx words/flushes）与统计；`pfss_layer_planner` 汇总跨 phase 预算（fail-closed）；`open_collector` 管理批量 Open；`phase_executor.hpp` 按 Need(Open/PfssCoeff/PfssTrunc) 驱动 flush_eval + finalize。
- **任务层**：`phase_tasks.hpp` 包含 `TruncTask`（faithful/GapARS/AutoTrunc）、`CubicPolyTask`（SiLU/nExp/Recip 3 mul + 2 trunc，必要时走参考路径）、`RsqrtTask`（仿射初值 + NR）、`LayerNormTask`（mean/var trunc + rsqrt + affine，ReferenceBackend 下可明文调试）；`runtime/staged_executor.hpp` 支持“两阶段”收集 PFSS/Open→单次 flush→finalize，`nn/softmax_block_task_staged.hpp` 演示跨任务共享一次 PFSS flush。
- **图/NN 路径**：`layer_graph` 维护 Rescale/Trunc/Clamp 节点和范围事实，hoist 可跨 rescale 链、add/sub/bias/Hadamard/mul_const/axpy；`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差，显式 rescale/trunc，无本地 shift），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划、GapCert、PfssSuperBatch。

## PhaseExecutor 状态机与执行路径

- PhaseExecutor 维护任务队列与 Need(Open/PfssCoeff/PfssTrunc/Done) 状态：NeedOpen 时 flush `OpenCollector`；NeedPfssCoeff/NeedPfssTrunc 时驱动对应 PFSS batch flush_eval + finalize，随后回到执行队列直到 Done。
- `PfssSuperBatch` 合并同类 job，记录 job/flush/opened_words 统计；`finalize_pfss_once`/`PfssPhasePlanner` 能在 phase 层面一次性 snapshot + flush（attention/MLP/softmax 波次已接入），默认更紧 budgets；`PfssLayerPlanner` 对多个 phase 的 PFSS 使用做聚合与限额，并提供层末尾安全 flush 钩子；`PfssAsyncRunner` 提供可选异步 flush/finalize 包装；`StagedExecutor` 演示 softmax 场景下收集多个任务后仅触发一次 PFSS flush 并再 finalize。
- LayerContext 保存 rescale/trunc 计划、GapCert 与共享的 PfssSuperBatch/LayerPlanner，便于跨 task hoist、范围决策与复用 planner 资源。

## 构建与运行

- **纯明文/快速验证（默认）**
  ```bash
  cmake -S . -B build
  cmake --build build
  ./build/sim_harness            # 断言式，两方模拟，ReluARS/GeLU 各 2000 轮
  ./build/test_suf_ref_eval      # 参考语义
  ./build/test_mask_rewrite      # 掩码重写性质
  ./build/test_compile_pfss      # SUF→PFSS 编译一致性
  ./build/test_sigmafast         # SigmaFast packed compare + interval LUT
  ./build/test_composite_runtime # 组合式 new 门运行时（Clear 后端，两方线程）
  ```
- **接入真实 myl7/fss（自动 FetchContent，禁用 CUDA/样例/测试）**
  ```bash
  cmake -S . -B build_myl7 -DSUF_FETCH_MYL7_FSS=ON -DSUF_USE_MYL7_FSS=ON
  cmake --build build_myl7
  ./build_myl7/sim_harness
  ```
  需要系统 `libsodium` + OpenMP，CMake 会下载 `myl7/fss@v0.7.1` 并仅构建 `dcf/dpf/cw_mac_bytes` 静态库；`myl7_fss_backend` 将 payload 自动填充到 `kLambda`（默认 16B）块并清空 MSB 符号位，key 中的 party 位选择正确种子，eval 会解析 header 并循环调用 `dcf_eval` 复原原始 payload 长度。

## 代码与数据对齐要点

- **Tape/顺序**：`proto/tape.hpp` 统一标签+长度，ReluARS/GeLU 的 Tape 顺序固定；在线 evaluator 严格按顺序消费，避免乱序泄露。
- **wrap bit**：所有 wrap/符号位均为加法 share（`u64`），Tape 存 share 值；在线通过 `BitRingOps::SEL(shared_wrap, nowrap_branch, wrap_branch)` 合成，不暴露公共位。
- **PFSS key/输出格式**：`pred` 采用 `kU64PerBit`（每比较一字），`coeff` Step-DCF 输出 `out_words = r*(d+1)`；`myl7_fss_backend` key header `[in_bits][num_chunks][party][0][payload_len_le32]`，chunk=`seed||cws||cw_np1`。
- **范围驱动 Rescale/Trunc/Clamp**：`range_analysis.hpp` 传播 `TensorFacts`，`clamp_range`/`kClamp`/`record_clamp` 标注 LN/SiLU/nExp/Recip/softmax/GeLU 的输出界，Bias add 用精确 min/max；`rescale_pass.cpp` hoist rescale over rescale chain、add/sub/bias/Hadamard/mul_const/axpy（需 frac/sign 匹配）；`MatmulRescaleSite` 携带 accum_range/prefer_gapars，GapARS 选择优先用于有 GapCert 的累加与 softmax/LN/recip 的 AutoTrunc。
- **PFSS 合并/flush**：trunc 通过 SUF builder 走 `enqueue_composite`，与 coeff 共用 PfssSuperBatch；`finalize_pfss_once`/`PfssPhasePlanner` 能在 phase 内强制单次 flush 并统计预算，attention/MLP/softmax 波次已接入；`PfssLayerPlanner` 可在层末尾做安全 flush 汇总；`PfssAsyncRunner` 提供可选异步 flush 包装（默认同步）。

## 当前状态 / 里程碑梳理

- **Milestone 1-8**：按 markdown 设计落地 SUF 语义、掩码重写、PFSS 编译、后端抽象、组合运行时，参考/自测均通过。
- **Milestone 11 已完成**  
  - 图/IR：linops、matmul、attention、MLP 去内联移位，统一显式 Rescale/TruncChoice/Clamp，范围信息贯穿 hoist/rescale 记录。  
  - 新任务：`LayerNormTask`（mean/var trunc + rsqrt NR + 行广播 mul + affine）、`RsqrtTask`（PFSS 仿射初值 + NR）、`CubicPolyTask`（SiLU/nExp/Recip 两次 trunc Horner），均接入 PhaseExecutor/PfssSuperBatch，可切 ReferenceBackend 调试。  
  - SUF/PFSS：SiLU/nExp/Recip 构造修补非单调区间，coeff payload 零 `r_in`；trunc 走 SUF 路径并复用 composite flush；AutoTrunc 依据 range_hint 选择 GapARS。  
  - Rescale/hoist：hoist 覆盖 rescale 链、add/sub/bias/Hadamard/mul_const/axpy，MatmulRescaleSite/softmax/LN/recip 带 range_hint/GapCert 优先 GapARS；公共 bias/residual 具备 GapCert。  
  - 执行层：attention/MLP/softmax phase 绑定 `PfssPhasePlanner`/`finalize_pfss_once` 单次 PFSS flush，默认更紧预算+统计。  
  - Transformer：attention/MLP/LN 全部任务化，无本地 shift；残差加法使用 `proto::add_mod`；小型 transformer+LN 回归用例通过。
  - 近期：Recip trunc 使用 clamped 证明界，attention out-proj 在拥有 qkv 证明界时也推导 proof abs→GapARS；BiasAdd 支持上移 rescale（溢出防护）以复用一次 trunc；异步 stress 覆盖 coeff+trunc 批次、planner 总计同步校验。
- **Milestone 11 待办/保留风险**：PhaseExecutor 级别的跨 phase/layer super-plan 与更激进的 hoist 仍缺；GapCert 证明保守，packing/flush 预算可再收紧；LN/激活 clamp 可以继续缩紧；性能向（GPU/SigmaFast PRG/Beaver 缓存）尚未展开。
- **Staged/super-plan 原型**：`StagedExecutor` + `test_staged_softmax` 展示 softmax 的 nExp/Recip/Trunc 可单次 PFSS flush，尚未推广到其他任务或跨 phase super-plan/异步合并。

## 测试现状

- `ctest`（build_ninja）：`test_softmax_task_{correctness,flush_counts}`、`test_trunc_task`、`test_layernorm_task`、`test_matmul_and_attention_executor`、`test_matmul_executor`、`test_staged_softmax` 通过。
- 其他可手动运行的二进制在 `build_ninja/`（如 `test_compile_pfss`、`test_mask_rewrite`、`test_sigmafast`、`test_composite_runtime`、`sim_harness` 等），便于针对 PFSS/掩码/组合运行时做深入回归。

## 核心文件与逻辑脉络

- **SUF 层**：`suf/*.hpp`（IR、BoolExpr、ref_eval、validate、mask_rewrite），`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译层**：`compiler/suf_to_pfss.cpp`（谓词去重 + wrap 重写 → Pred/Coeff ProgramDesc），`compiled_suf_gate.hpp` 保存掩码/布局，`pfss_program_desc.hpp`/`range_analysis.hpp` 描述 PFSS 程序与范围/GapCert 工具。
- **后端/组合运行时**：`gates/composite_fss.hpp`（CompositeKeyPair，r_in/out/wrap share、Beaver、compiled gate），`runtime/pfss_superbatch`、`open_collector`、`phase_executor.hpp` 聚合 PFSS coeff/trunc/open 请求并按 Need(Open/PfssCoeff/PfssTrunc) 状态机 flush/finalize；`pfss_phase_planner.hpp` 记录预算/统计并可单次 flush；`runtime/staged_executor.hpp` 演示一次 flush 覆盖多任务的 prepare/finalize 流程。
- **任务状态机（`include/runtime/phase_tasks.hpp`）**：  
  - `TruncTask`：open masked x̂ → PFSS trunc（faithful/GapARS/AutoTrunc）。  
  - `CubicPolyTask`：open x̂ → PFSS 取 coeff → 3×Beaver mul + 2×Trunc（SiLU/nExp/Recip）或 reference spec。  
  - `RsqrtTask`：PFSS 仿射初值 + NR（3 mul/iter + trunc）。  
  - `LayerNormTask`：mean/var trunc + rsqrt 迭代 + 行广播 mul + affine（ReferenceBackend 可明文调试）。  
  - `PhaseExecutor`：多任务循环，驱动 Open/coeff/trunc flush_eval + finalize，记录 flush/job/opened_words/规划统计。
- **NN 路径**：`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划、GapCert 与 PfssSuperBatch。

## 当前难点与风险

- **GapCert 证明不足**：范围/GapCert 仍靠启发式 clamp，未引入形式化 gap_cert 证明链，限制了更激进的 hoist 与 AutoTrunc 选择；刚加入了 Proof/Hint 占位（TensorFacts.abs_kind/gap）但尚未贯通全链路。
- **Super-plan 受限**：现有 planner 仅做 phase/layer 内的单次 flush + 预算检查，未做跨 phase/layer 合并或 causal-mask 稀疏批次优化；packing 边界仅在 planner_causal 中做简单断言，PhaseExecutor 仅在 keep_batches=false 时清 batch（已支持跨 phase 保留，但还未添加“stall-driven flush”策略）。
- **Async 覆盖有限**：异步 PFSS 仅在存在独立 PFSS 通道时启用，且只在层末集中 flush；缺少更细粒度的异步/overlap 评估。
- **性能工作未展开**：SigmaFast PRG/packing、Beaver/三元组缓存、GPU/SoA 批处理等性能向优化尚未启动。

## 后续建议

1. 推进 PFSS super-plan（跨 phase 预分组、单次 flush）与 packing/flush 预算收紧。  
2. 加强 gap_cert 证明与范围夹紧，进一步驱动 GapARS/hoist，并覆盖更多算子。  
3. 性能化：SigmaFast PRG/packing 实现、Beaver/三元组缓存、GPU/SoA 批处理与 overlap。

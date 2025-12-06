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
  runtime/                        # PhaseExecutor、PhaseTask 状态机、PfssSuperBatch、OpenCollector、PfssPhasePlanner
  nn/                             # attention/MLP/transformer 层，LayerContext/hoist/rescale/trunc 记录
  proto/                          # bits-in/bytes-out DCF 层（clear/myl7/sigmafast 后端、Beaver、tape）
src/compiler/suf_to_pfss.cpp      # SUF→PFSS 编译实现
src/runtime/*                     # PfssSuperBatch/OpenCollector 实现
src/nn/*                          # attention_block / mlp_block / transformer_layer
src/demo/sim_harness.cpp          # 全流程两方模拟 + 断言
src/demo/test_*                   # SUF/mask/编译/任务 测试
expand*.md, initial.md, milestone*.md, advice*.md, revise_m11_*.md, paper.md
```

## 核心流程与逻辑

- **SUF 层**：`suf/validate.hpp` 校验区间/度数，`suf/ref_eval.hpp` 提供语义金标；`suf/mask_rewrite.hpp` 将谓词迁移到 `hatx=x+r` 域并引入 wrap bit（加法 share）；`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译管线**：`compiler/suf_to_pfss.cpp` 收集原始谓词→掩码重写→去重→Pred/Coeff ProgramDesc（Step-DCF 或 Interval LUT，处理 wrap 分段），输出 `CompiledSUFGate`（掩码、布局、额外 gate_kind/spec）。参考测试 `test_compile_pfss.cpp` 对比 `ref_eval`。
- **范围与 GapARS**：`compiler/pfss_program_desc.hpp` 定义 `RangeInterval`；`GateParams::range_hint`、`TensorFacts::gap_cert` 贯穿图。`GateKind::AutoTrunc` 在 `truncation_lowering` 中由 `select_trunc_kind(range_hint, frac_bits)` 自动挑选 GapARS/faithful。Matmul/softmax/recip/LN/MLP builder 会填充 range_hint，`MatmulRescaleSite` 附带 accum_range/prefer_gapars，hoist 与 trunc 计划优先 GapARS 当 `has_gap_cert`。
- **PFSS/组合运行时**：`runtime/pfss_superbatch` 聚合 coeff+trunc（trunc 也走 `enqueue_composite`）；`open_collector` 管理批量 Open；`phase_executor.hpp` 驱动任务并按 Need(Open/PfssCoeff/PfssTrunc) flush/finalize；`finalize_pfss_once`/`pfss_phase_planner.hpp` 支持 phase 内单次 flush（attention softmax/QKV/out-proj、MLP 波次已启用）。
- **任务层**：`phase_tasks.hpp` 包含 `TruncTask`（faithful/GapARS/AutoTrunc）、`CubicPolyTask`（SiLU/nExp/Recip 3 mul + 2 trunc）、`RsqrtTask`（仿射初值 + NR）、`LayerNormTask`（mean/var trunc + rsqrt + affine，ReferenceBackend 下可明文调试）。
- **图/NN 路径**：`layer_graph` 维护 Rescale/Trunc 节点和范围事实，hoist 可跨 rescale 链、add/sub/bias/Hadamard/mul_const/axpy；`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差，显式 rescale/trunc，无本地 shift），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划与 PfssSuperBatch。

## PhaseExecutor 状态机与执行路径

- PhaseExecutor 维护任务队列与 Need(Open/PfssCoeff/PfssTrunc/Done) 状态。遇到 NeedOpen 时 flush `OpenCollector`；NeedPfssCoeff/NeedPfssTrunc 时驱动对应 PFSS batch flush_eval + finalize；任务继续执行直到 Done。
- `PfssSuperBatch` 将同类 job 合并并支持 stats（job_count/flush_count/opened_words）。`finalize_pfss_once` 确保一个 phase 只 flush 一次（若调用），`PfssPhasePlanner` 可以 snapshot 并强制单次 flush。
- LayerContext 记录 rescale/trunc 计划、gap_cert、PfssSuperBatch 共享，便于跨 task hoist 和范围决策。

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
- **范围驱动 Rescale/Trunc**：`range_analysis.hpp` 传播 `TensorFacts`（range、gap_cert、frac_bits、is_signed）；`rescale_pass.cpp` hoist rescale over rescale chain、add/sub/bias/Hadamard/mul_const/axpy（需 frac/sign 匹配）；`MatmulRescaleSite` 携带 accum_range/prefer_gapars，GapARS 选择优先用于有 gap_cert 的 accum 与 softmax/LN/recip 的 AutoTrunc。
- **PFSS 合并/flush**：trunc 通过 SUF builder 走 `enqueue_composite`，与 coeff 共用 PfssSuperBatch；`finalize_pfss_once`/`PfssPhasePlanner` 能在 phase 内强制单次 flush，attention/MLP/softmax 波次已接入。

## 当前状态 / 里程碑梳理

- **Milestone 1-8**：SUF 语义、掩码重写、PFSS 编译、后端抽象、组合运行时、参考测试全部落地，核心自测与 harness 通过。
- **Milestone 11 已完成**  
  - 图/IR：linops、matmul、attention、MLP 去内联移位，统一显式 Rescale/TruncChoice，范围信息贯穿 hoist/rescale 记录。  
  - 新任务：`LayerNormTask`（mean/var trunc + rsqrt NR + 行广播 mul + affine）、`RsqrtTask`（PFSS 仿射初值 + NR）、`CubicPolyTask`（SiLU/nExp/Recip 两次 trunc Horner），均接入 PhaseExecutor/PfssSuperBatch。  
  - SUF/PFSS：SiLU/nExp/Recip 构造修补非单调区间，coeff payload 零 `r_in`；trunc 走 SUF 路径并复用 composite flush；AutoTrunc 依据 range_hint 选择 GapARS。  
  - Rescale/hoist：hoist 覆盖 rescale 链、add/sub/bias/Hadamard/mul_const/axpy，MatmulRescaleSite/softmax/LN/recip 带 range_hint/gap_cert 优先 GapARS。  
  - 执行层：attention/MLP/softmax phase 使用 `finalize_pfss_once` 单次 PFSS flush；PfssSuperBatch 统计 packing/flush。  
  - Transformer：attention/MLP/LN 全部任务化，无本地 shift；残差加法使用 `proto::add_mod`；小型 transformer+LN 回归用例通过。
- **Milestone 11 待办/保留风险**：PhaseExecutor 级别的全局 super-plan（跨 phase 预分组 PFSS）尚未完成；gap_cert 证明仍保守，未做更激进的 rescale/trunc hoist；packing/flush 预算可再收紧；LN/激活 clamp 的范围可继续强化；GPU overlap/吞吐优化未动。

## 测试现状

- `ctest`（build_ninja）：`test_softmax_task_{correctness,flush_counts}`、`test_trunc_task`、`test_layernorm_task`、`test_matmul_and_attention_executor`、`test_matmul_executor` 均通过。
- 额外：`test_compile_pfss`、`test_mask_rewrite`、`test_sigmafast`、`test_composite_runtime`、`sim_harness`（ReluARS/GeLU 2000 轮）通过。

## 核心文件与逻辑脉络

- **SUF 层**：`suf/*.hpp`（IR、BoolExpr、ref_eval、validate、mask_rewrite），`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译层**：`compiler/suf_to_pfss.cpp`（谓词去重 + wrap 重写 → Pred/Coeff ProgramDesc），`compiled_suf_gate.hpp` 保存掩码/布局，`pfss_program_desc.hpp` 描述 PFSS 程序。
- **后端/组合运行时**：`gates/composite_fss.hpp`（CompositeKeyPair，r_in/out/wrap share、Beaver、compiled gate），`runtime/pfss_superbatch`、`open_collector`、`phase_executor.hpp` 聚合 PFSS coeff/trunc/open 请求并按 Need(Open/PfssCoeff/PfssTrunc) 状态机 flush/finalize，`pfss_phase_planner.hpp` 支持 phase 内单次 flush。
- **任务状态机（`include/runtime/phase_tasks.hpp`）**：  
  - `TruncTask`：open masked x̂ → PFSS trunc（faithful/GapARS/AutoTrunc）。  
  - `CubicPolyTask`：open x̂ → PFSS 取 coeff → 3×Beaver mul + 2×Trunc（SiLU/nExp/Recip）或 reference spec。  
  - `RsqrtTask`：PFSS 仿射初值 + NR（3 mul/iter + trunc）。  
  - `LayerNormTask`：mean/var trunc + rsqrt 迭代 + 行广播 mul + affine（ReferenceBackend 可明文调试）。  
  - `PhaseExecutor`：多任务循环，驱动 Open/coeff/trunc flush_eval + finalize，记录 flush/job/opened_words 统计。
- **NN 路径**：`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划与 PfssSuperBatch。

## 后续建议

1. 推进 PFSS super-plan（跨 phase 预分组、单次 flush）与 packing/flush 预算收紧。  
2. 加强 gap_cert 证明与范围夹紧，进一步驱动 GapARS/hoist，并覆盖更多算子。  
3. 性能化：SigmaFast PRG/packing 实现、Beaver/三元组缓存、GPU/SoA 批处理与 overlap。

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
  runtime/                        # PhaseExecutor、PhaseTask 状态机、PfssSuperBatch、OpenCollector
  nn/                             # attention/MLP/transformer 层，LayerContext/hoist/rescale 记录
  proto/                          # bits-in/bytes-out DCF 层（clear/myl7/sigmafast 后端、Beaver、tape）
src/compiler/suf_to_pfss.cpp      # SUF→PFSS 编译实现
src/runtime/*                     # PfssSuperBatch/OpenCollector 实现
src/nn/*                          # attention_block / mlp_block / transformer_layer
src/demo/sim_harness.cpp          # 全流程两方模拟 + 断言
src/demo/test_*                   # SUF/mask/编译/任务 测试
expand*.md, initial.md, milestone*.md, advice*.md, revise_m11_*.md, paper.md
```

## 核心流程与逻辑

- **SUF 层**：`suf/validate.hpp` 校验区间/度数，`suf/ref_eval.hpp` 提供语义金标；`suf/mask_rewrite.hpp` 将谓词迁移到 `hatx=x+r` 域并引入 wrap bit（加法 share），`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译管线**：`compiler/suf_to_pfss.cpp` 收集原始谓词→掩码重写→去重→Pred/Coeff ProgramDesc（Step-DCF 或 Interval LUT，处理 wrap 分段），输出 `CompiledSUFGate`（掩码、布局、额外 gate_kind/spec）。参考测试 `test_compile_pfss.cpp` 对比 `ref_eval`。
- **后端抽象**：`proto/backend_clear.hpp`（明文）、`proto/myl7_fss_backend.hpp`（自动分块 DCF/party header）、`proto/sigma_fast_backend_ext.hpp`（packed compare/LUT stub），BeaverMul64/BitRingOps、tape/pack_utils 辅助。
- **组合运行时/状态机**：`runtime/pfss_superbatch` 聚合 PFSS coeff/trunc 请求，`open_collector` 管理批量 Open，`phase_executor.hpp` 驱动任务并按 Need(Open/PfssCoeff/PfssTrunc) flush/finalize。
  - `TruncTask`：open masked x̂ → PFSS trunc（faithful/GapARS）。
  - `CubicPolyTask`：open x̂ → PFSS coeff → 3×Beaver mul + 2×Trunc（SiLU/nExp/Recip），可走 reference spec。
  - `RsqrtTask`：PFSS 仿射初值 + NR（3 mul/iter + trunc）。
  - `LayerNormTask`：mean/var trunc + rsqrt 迭代 + 行广播 mul + affine；ReferenceBackend 下可直接明文调试。
- **图/NN 路径**：`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差，显式 rescale/trunc，无本地 shift），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划与 PfssSuperBatch。范围事实贯穿记录，用于选择 trunc kind 与 rescale。

## 构建与运行

- **纯明文/快速验证（默认）**
  ```bash
  cmake -S . -B build
  cmake --build build
  ./build/sim_harness            # 断言式，两方模拟，ReluARS/GeLU 各 2000 轮
  ./build/test_suf_ref_eval      # new 参考语义
  ./build/test_mask_rewrite      # 掩码重写性质
  ./build/test_compile_pfss      # new→PFSS 编译一致性
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

- **Tape/顺序**：`proto/tape.hpp` 统一标签+长度，ReluARS/GeLU 的 Tape 顺序在 README 注释与代码中固定；在线 evaluator 严格按顺序消费，避免乱序泄露。
- **wrap bit**：所有 wrap/符号位均为加法 share（`u64`），Tape 存 share 值；在线通过 `BitRingOps::SEL(shared_wrap, nowrap_branch, wrap_branch)` 合成，不再暴露公共位。
- **PFSS key/输出格式**：`pred` 采用 `kU64PerBit`（每比较一字），`coeff` Step-DCF 输出 `out_words = r*(d+1)`；`myl7_fss_backend` key header `[in_bits][num_chunks][party][0][payload_len_le32]`，chunk=`seed||cws||cw_np1`。
- **SUF 编译结果**：`CompiledSUFGate` 持有 per-piece BoolExpr（变量索引=raw query idx，wrap bits 追加在末尾），并保留掩码后的系数分段/切点，使后端/评估器无需了解原始掩码。

## 状态与下一步

- Milestone 1-6 ✅：核心 runtime + Tape + SUF 语义/掩码重写 + SUF→PFSS 编译 + 清晰的后端接口；wrap 泄露已修复（additive share + MPC SEL）。`sim_harness`、`test_*` 皆通过（ReluARS/GeLU 断言式随机 2000 例）。
- 已接入 myl7/fss：CMake 可一键拉取/链接，后端实际调用 `dcf_gen/eval`，支持 payload 分块；未检测到头文件时自动回退到内存 stub。
- SigmaFast（CPU）现有 packed compare/interval LUT stub，输出 party0 掩码或 payload，配合 `test_sigmafast` 做正确性；后续可替换为 SIGMA 风格 PRG/packing 以获得真实吞吐。
- 组合式 SUF 运行时雏形：`gates/composite_fss.hpp` + `test_composite_runtime`（使用 ClearBackend+Beaver/线程通道）；接口已为通用后端设计，Myl7/SigmaFast 需要按各自输出域做比特重分享/布尔 DAG MPC。
- 待优化：真实 Delta/LUT/样条系数替换 toy 数据；性能化批处理（GPU/SoA）与 SigmaFast PRG 优化。
- Milestone 11 进展：linops/matmul/attention/MLP 去内联移位，显式 Rescale/TruncChoice；LayerNorm/Rsqrt/Softmax/SiLU/nExp/Recip 任务化并接入 PhaseExecutor；SiLU/nExp/Recip SUF 单调修复、零 r_in；小型 transformer+LN 回归通过。剩余：全图 rescale-hoist 收束、范围驱动 trunc 选择、packing/flush 计数约束、LN 范围绑定，必要时收紧 demo 容差（当前容忍 ±1 LSB）。

## 测试现状

- `ctest`（build）：`test_softmax_task_{correctness,flush_counts}`、`test_trunc_task`、`test_layernorm_task`、`test_matmul_and_attention_executor`、`test_matmul_executor` 均通过。
- 额外：`test_compile_pfss`、`test_mask_rewrite`、`test_sigmafast`、`test_composite_runtime`、`sim_harness`（ReluARS/GeLU 2000 轮）通过。


## 当前状态 / 里程碑梳理

- **Milestone 1-8**：SUF 语义、掩码重写、PFSS 编译、后端抽象、组合运行时全部落地，核心自测与 harness 通过。
- **Milestone 11 已完成**  
  - 图/IR：linops、matmul、attention、MLP 去除内联移位，统一显式 Rescale/TruncChoice，范围信息贯穿 hoist/rescale 记录。  
  - 新任务：`LayerNormTask`（mean/var trunc + rsqrt NR + 行广播 mul + affine）、`RsqrtTask`（PFSS 仿射初值 + NR）、`CubicPolyTask`（SiLU/nExp/Recip 两次 trunc Horner）、均接入 PhaseExecutor/PfssSuperBatch。  
  - SUF/PFSS：SiLU/nExp/Recip 构造修补非单调区间，coeff payload 统一零 `r_in`；`CubicPolyTask` reference 路径直连 `ref_silu_fixed/ref_nexp_fixed`。  
  - Transformer：attention/MLP/LN 全部任务化，无本地 shift；残差加法使用 `proto::add_mod`；小型 transformer+LN 回归用例通过。  
- **Milestone 11 待办**：全图 rescale-hoist 进一步贯穿，范围事实馈入 trunc 选择，全局 packing/flush 计数与约束，LN 范围绑定到图元数据，必要时收紧 demo 容差（当前 demo 容忍 ±1 LSB）。

## 核心文件与逻辑脉络

- **SUF 层**：`suf/*.hpp`（IR、BoolExpr、ref_eval、validate、mask_rewrite），`suf/suf_silu_builders.hpp` 生成 SiLU/nExp/Recip 样条 SUF（r_out=4 coeff，degree=0，零 r_in）。
- **编译层**：`compiler/suf_to_pfss.cpp`（谓词去重 + wrap 重写 → Pred/Coeff ProgramDesc），`compiled_suf_gate.hpp` 保存掩码/布局，`pfss_program_desc.hpp` 描述 PFSS 程序。
- **后端/组合运行时**：`gates/composite_fss.hpp`（CompositeKeyPair，r_in/out/wrap share、Beaver、compiled gate），`runtime/pfss_superbatch`、`open_collector`、`phase_executor.hpp` 聚合 PFSS coeff/trunc/open 请求并按 Need(Open/PfssCoeff/PfssTrunc) 状态机 flush/finalize。
- **任务状态机（`include/runtime/phase_tasks.hpp`）**：  
  - `TruncTask`：open masked x̂ → PFSS trunc（faithful/GapARS）。  
  - `CubicPolyTask`：open x̂ → PFSS 取 coeff → 3×Beaver mul + 2×Trunc（SiLU/nExp/Recip）或 reference spec。  
  - `RsqrtTask`：PFSS 仿射初值 + NR（3 mul/iter + trunc）。  
  - `LayerNormTask`：mean/var trunc + rsqrt 迭代 + 行广播 mul + affine（可在 ReferenceBackend 下直接明文重构，便于调试）。  
  - `PhaseExecutor`：多任务循环，驱动 Open/coeff/trunc flush_eval + finalize，记录 flush/job/opened_words 统计。
- **NN 路径**：`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc 计划 + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU CubicPolyTask），`nn/transformer_layer.cpp`（两次 LayerNormTask + attention + MLP + 残差），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划与 PfssSuperBatch。

## 测试与当前结果

- `ctest`（`build`）覆盖：`test_softmax_task_{correctness,flush_counts}`、`test_trunc_task`、`test_layernorm_task`、`test_matmul_and_attention_executor`、`test_matmul_executor` —— **全部通过**。
- 重要补充：`test_compile_pfss`、`test_mask_rewrite`、`test_sigmafast`、`test_composite_runtime`、`sim_harness`（ReluARS/GeLU 随机 2000 轮）均通过。

## 后续建议

1. 若需 bit-for-bit，收紧 demo 容差并沿 LN/rsqrt/attention 追踪舍入差异。  
2. 继续 Milestone 11 剩余项：全图 rescale-hoist、范围驱动 trunc 选择、packing/flush 计数约束、LN 范围绑定。  
3. 性能化：SigmaFast PRG/packing 真实实现、Beaver/三元组缓存与 GPU/SoA 批处理。

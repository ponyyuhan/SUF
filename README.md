# SUF/PFSS Prototype

## 目录结构（精简）

```
CMakeLists.txt          # 可选自动拉取 myl7/fss，CUDA 架构可配
include/                # 所有头文件
  suf/                  # SUF IR、谓词、BoolExpr、mask rewrite + ref_eval/validate
  compiler/             # PFSS 描述/编译产物、range/gap、suf_to_pfss.cpp
  gates/                # 组合门 API（SiLU/Recip/nExp/Softmax/Trunc 等）
  runtime/              # PhaseExecutor、PfssSuperBatch、OpenCollector、planner、staged/async 执行
  nn/                   # attention/MLP/transformer 层、LayerContext、softmax 任务
  proto/                # 后端抽象（clear/myl7/sigmafast/GPU）、Beaver、tape
  mpc/core/pfss/...     # 环运算、pack/unpack、通道抽象
cuda/                   # GPU 后端（AES-CTR、DPF、packed predicates/LUT、matmul）
src/                    # 运行时/NN 实现、bench、demo/tests
docs/                   # milestone/设计文档
```

## 核心流程与逻辑

- **SUF→PFSS 编译**：`compiler/suf_to_pfss.cpp` 收集谓词、掩码重写、去重，生成 `Pred/Coeff ProgramDesc`（Step-DCF 或 Interval LUT，处理 wrap 段），输出 `CompiledSUFGate`（掩码、布局、gate_kind）。`test_compile_pfss.cpp` 对标 `ref_eval`。
- **范围/GAPARS**：`range_analysis.hpp` 传播 `TensorFacts`（range/gap/frac_bits/is_signed），`record_clamp`/`LayerGraph::kClamp` 标注 LN/SiLU/nExp/Recip/softmax 等已知界；`GateKind::AutoTrunc` 通过 `select_trunc_kind` 选 GapARS/faithful。
- **运行时/状态机**：`PhaseExecutor` 循环任务队列，根据 Need(Open/PfssCoeff/PfssTrunc) flush `OpenCollector` 与 `PfssSuperBatch`（coeff+trunc 同批合并），再 finalize。`PfssPhasePlanner` 支持单次 flush+统计（attention/MLP/softmax 波次已接入）；`PfssLayerPlanner` 聚合多 phase 预算；`PfssAsyncRunner` 可选异步 flush。`StagedExecutor` 演示 softmax 跨任务的一次 PFSS flush。
- **任务层**（`runtime/phase_tasks.hpp`）：`TruncTask`（faithful/GapARS/AutoTrunc），`CubicPolyTask`（SiLU/nExp/Recip 3 mul + 2 trunc），`RsqrtTask`（仿射初值 + NR），`LayerNormTask`（mean/var trunc + rsqrt + affine），均复用 PhaseExecutor/PfssSuperBatch。
- **NN 路径**：`nn/attention_block.cpp`（任务化 softmax/recip/nexp + matmul trunc + KV cache），`nn/mlp_block.cpp`（两段 matmul trunc + SiLU），`nn/transformer_layer.cpp`（两次 LayerNorm + attention + MLP + 残差），`nn/layer_context.hpp` 记录 hoist/rescale/trunc 计划、GapCert、PfssSuperBatch/GPU stager。
- **Packing hints**：`Pred/CoeffProgramDesc` 支持 `eff_bits`（默认 64）和 `ragged` 形状元数据；nExp/Recip keygen 会按 spec 自动给出 `eff_bits` hint，SigmaFast/GPU packed keygen 会使用更小的 `eff_bits`；ragged/causal 的 `row_offsets/valid_lens` 已在 softmax 任务与 planner 端贯穿，但具体 pack/scatter 仍由任务侧完成。

## PhaseExecutor 状态机与执行路径

- Need(Open/PfssCoeff/PfssTrunc/None) 驱动：遇到 NeedOpen flush `OpenCollector`，NeedPfss* 则 flush_eval + finalize 对应 `PfssSuperBatch`，直至任务 Done。
- `PfssSuperBatch`：按 key/布局分桶合并 job，统计 hatx/pending/jobs/flush；同一批同时支持 coeff+trunc；GPU stager 可选。
- `finalize_pfss_once`/`PfssPhasePlanner`：phase 内单次 flush+统计；`PfssLayerPlanner` 聚合多 phase 限额并提供层尾 barrier；`PfssAsyncRunner` 在有独立通道时可并行 flush。

## 构建与运行

- **CPU/默认**  
  ```bash
  cmake -S . -B build
  cmake --build build
  ./build/sim_harness
  ctest -R "test_(suf_ref_eval|mask_rewrite|compile_pfss|sigmafast|composite_runtime|softmax_task|trunc_task|layernorm_task|matmul_and_attention_executor|staged_softmax)" -V
  ```
- **myl7/fss 后端**（自动 FetchContent，禁用其 CUDA/样例/测试）  
  ```bash
  cmake -S . -B build_myl7 -DSUF_USE_MYL7_FSS=ON -DSUF_FETCH_MYL7_FSS=ON
  cmake --build build_myl7
  ./build_myl7/sim_harness
  ```
- **CUDA/Packed 路径**（需要 nvcc+GPU，默认架构 sm80/86，可通过 `CMAKE_CUDA_ARCHITECTURES` 覆盖）  
  ```bash
  cmake -S . -B build_cuda -DSUF_ENABLE_CUDA=ON
  cmake --build build_cuda -j
  ctest -R "test_cuda_(packed_pfss|pred_mask|prg|pack_effbits|softmax_gpu_smoke|pfss_gpu)" -V
  RUN_GPU_COMPOSITE=1 ctest -R test_pfss_gpu -V   # 开启 GPU packed composite
  ./build_cuda/bench_gemm_overlap                 # PFSS+GEMM 重叠基准
  ```
- **流/重叠**：PFSS GPU backend 暴露 compute stream，GEMM 走独立非阻塞流（matmul_default_stream）；PFSS block size 可用 `SUF_PFSS_GPU_BLOCK` 调节；matmul tiling 可用 `SUF_MATMUL_GPU_TILE=wide|narrow` 控制。
- **缓存/基准**：GPU PFSS 默认缓存 keys/hatx（`SUF_NO_CACHE_KEYS`/`SUF_NO_CACHE_HATX` 关闭）；`bench_gemm_overlap`、`bench_pfss_cpu_gpu`、`bench_softmax_norm`（`--preset=safe`，`SUF_BENCH_DEVICE_TIME=1` 记录设备时间）可用于对比 CPU/GPU 与重叠收益。

## 代码与数据对齐要点

- Tape 顺序固定（ReluARS/GeLU），在线 evaluator 严格按序消费；trunc/ARS 的 carry/sign/wrap 均为加法 share（u64），wrap 由 PFSS 谓词 `1[hatx<r_in]` 直接输出（不做 public `r_in` 比较）。
- PFSS key/输出：pred 默认 `kU64PerBit_Xor`，coeff Step-DCF 输出 `out_words=r*(d+1)`；SigmaFast packed key 带 thresholds+AES round_keys。
- 范围驱动 rescale/trunc/clamp：range/gap_hint 贯穿 graph 与 trunc 降级，MatmulRescaleSite/softmax/LN/recip 优先 GapARS；Bias/residual 携带 GapCert。
- PFSS 合并/预算：coeff+trunc 共用 PfssSuperBatch，planner 统计 jobs/hatx/flush，limits fail-closed；GPU 路径可设置 device-byte 预算。

## 当前状态 / 里程碑梳理

- Milestone 1-8：SUF 语义、掩码重写、PFSS 编译/后端、组合运行时、任务化 softmax/LN/MLP/attention，全套 CPU 测试通过。
- Milestone 11（GPU PFSS 验证/overlap）  
  - GPU 后端：AES-CTR PRG 修正，packed CDPF/vector-DPF（pred bitmask、interval LUT payload），device key 缓存，staged eval 接口曝光 compute stream。  
  - Packed/bitmask：GPU 支持 packed predicates/cuts，eff_bits 打包/解包（`test_cuda_pack_effbits`）；ragged/causal 形状元数据与 bytes 回归（`test_planner_causal_bytes`）已接入，pack/scatter 仍在任务侧。  
- Overlap：LayerContext 暴露 PFSS compute stream，GPU matmul（BK=32/64 自适应，vec load，可用 `SUF_MATMUL_GPU_TILE=wide|narrow` 强制）走独立非阻塞流，PFSS kernel block 可调 `SUF_PFSS_GPU_BLOCK`，避免与 PFSS 流串行；`bench_gemm_overlap` 同时跑 PFSS+GEMM 并输出 PFSS/GEMM/overlap 三组计时（本机参考：PFSS≈9.99 ms、GEMM≈5.14 ms、overlap≈10.02 ms）。  
  - 软/硬回归：`test_cuda_prg`、`test_cuda_packed_pfss`、`test_cuda_pred_mask`、`test_pfss_gpu`（可开 RUN_GPU_COMPOSITE）、`test_softmax_gpu_smoke`、GapARS/Faithful trunc CUDA 等效。
- 基准/对比：新增基准脚本与配置（见下）。

## 测试现状

- CPU 路径：上述 ctest 套件稳定；`sim_harness`/`test_composite_runtime`/`test_sigmafast`/`test_mask_rewrite` 等用于语义回归。
- CUDA 路径：`test_cuda_prg`、`test_cuda_packed_pfss`、`test_cuda_pack_effbits`、`test_pfss_gpu`、`test_softmax_gpu_smoke` 通过（需 GPU）；`bench_gemm_overlap` 提供 PFSS/GEMM 重叠计时。

## 核心文件与逻辑脉络

- 编译/IR：`compiler/suf_to_pfss.cpp`，`compiler/compiled_suf_gate.hpp`，`compiler/range_analysis.hpp`，`compiler/truncation_lowering.hpp`。
- 运行时：`runtime/pfss_superbatch.cpp`，`runtime/phase_executor.hpp`，`runtime/pfss_phase_planner.hpp`，`runtime/open_collector.cpp`，`runtime/pfss_async_runner.hpp`，`runtime/staged_executor.hpp`。
- 任务：`runtime/phase_tasks.hpp`（Trunc/CubicPoly/Rsqrt/LayerNorm），`nn/softmax_block_task(_staged).hpp`。
- 后端：`gates/composite_fss.hpp`（Composite key/eval），`proto/backend_*`（clear/myl7/sigmafast/gpu）。
- GPU：`cuda/pfss_kernels.cu`（AES-CTR、packed compare/LUT、eff_bits 解包），`cuda/pfss_backend_gpu.cu`（device key + staged eval），`src/nn/matmul_gpu.cu`（流感知 matmul），`src/bench/bench_gemm_overlap.cpp`。
- Bench/对标：`bench/run_sigma_vs_suf.py`（统一 orchestrator），`bench/configs/sigma_vs_suf.json`（模型/参数预设），`scripts/describe_hardware.py`（记录硬件 JSON），`scripts/build_sigma.sh`（辅助构建 Sigma），`src/demo/bench_suf_transformer.cpp`（单层 transformer forward 基准，输出 Sigma 兼容 JSON 日志）。

## 当前难点与风险

- GapCert/范围仍保守，限制更激进的 hoist 与 AutoTrunc 选择。
- Super-plan/packing 仍是 phase/layer 粒度，未做跨 phase 融合或更细的 stall/bytes 驱动 flush，causal/ragged 预算可进一步收紧。
- GPU 性能：matmul 仍为简化 tiling，未用 WMMA/半精度拆半；PFSS/GEMM overlap 需更稳的 stream/pipeline；Beaver/三元组/GPU 缓存策略尚浅。
- Ragged packing 仍主要由任务侧 pack/scatter（PFSS job 只携带元数据）；eff_bits 已在 nExp/Recip 等正域门上自动生效，其它门若需缩 bitwidth 仍要先裁剪不可达区间/提供安全的范围证明。

## 后续建议

1) 完善 super-plan 与 bytes/packing 预算（含 causal/ragged）、增加 planner 回归。  
2) 收紧 GapCert/abs 界并贯穿更多算子，提升 GapARS 覆盖面。  
3) GPU 性能：引入 WMMA/半精度拆半、SoA 打包、Beaver/三元组缓存；精炼 PFSS/GEMM pipeline（事件/双流）与更丰富基准；驱动 eff_bits/ragged packing 到 planner 与 PFSS pack/unpack。  
4) 完整对标 Sigma：扩展 `bench_suf_transformer` 到多层/真实权重（或加载 HF 权重），补全预处理时间/字节统计与更细的非线性计数，在统一 orchestrator 下产出 CSV/JSONL。

# SUF/PFSS Prototype (GPT-5 Codex Agent)

## 目录结构（精简）

```
CMakeLists.txt                    # 可选自动拉取 myl7/fss
include/
  core/                           # 环运算、pack/unpack
  mpc/                            # Beaver 三元组、通道抽象
  suf/                            # SUF IR、谓词、BoolExpr、mask rewrite + ref_eval/validate
  compiler/                       # PFSS 描述（pred/coeff）、编译产物、suf_to_pfss.cpp
  pfss/                           # 抽象 PFSS 接口
  gates/                          # 门级 API + ReLU/ReluARS/GeLU SUF 占位
  proto/                          # bits-in/bytes-out DCF 原型层
    backend_clear.hpp             # 明文 DCF/区间 LUT
    myl7_fss_backend.hpp          # 真实 myl7/fss 适配（支持 payload 分块）
    sigma_fast_backend_ext.hpp    # packed compare / interval LUT（CPU stub，支持 in_bits<=64 掩码打包）
    beaver*/bit_ring_ops.hpp      # 批量 Beaver + 布尔算子
    tape.hpp                      # Vec/File tape，统一标签格式
    reluars_* / gelu_*            # 离线 dealer + 在线 evaluator
    pfss_utils.hpp/pack_utils.hpp # eval 辅助、key 打包
src/compiler/suf_to_pfss.cpp      # SUF→PFSS 编译实现
src/demo/sim_harness.cpp          # 全流程两方模拟 + 断言
src/demo/test_*                   # SUF/mask/编译 测试
expand*.md, initial.md, milestone*.md, paper.md
```

## 核心流程与逻辑

- **SUF IR & 校验**：`suf/validate.hpp` 确认区间单调、度数/谓词参数合法；`suf/ref_eval.hpp` 作为语义金标准（谓词、BoolExpr、多项式、分段）。
- **掩码重写 (§3.3)**：`suf/mask_rewrite.hpp` 将 `x<β / x mod 2^f < γ / MSB(x+c)` 在 `hatx = x+r` 域下重写为公共阈值比较 + 秘密 wrap bit。wrap 以 **additive u64 share** 存储，在线用 MPC `SEL`。
- **SUF→PFSS 编译（Milestone 5/6）**：
  - `compiler/pfss_program_desc.hpp` 描述 `PredProgramDesc`（原始比较查询集合）与 `CoeffProgramDesc`（Step-DCF 或 Interval-LUT）。
  - `compiler/compiled_suf_gate.hpp` 存储掩码 `r_in/r_out[]`、编译后的 BoolExpr（变量=raw predicate idx + wrap bits）、系数模式元数据。
  - `compiler/suf_to_pfss.hpp` / `src/compiler/suf_to_pfss.cpp`：收集原始谓词→掩码重写→去重→生成谓词程序；系数区间按掩码旋转/拆 wrap → Step-DCF deltas 或 Interval LUT。
  - 参考测试：`src/demo/test_compile_pfss.cpp` 直接在明文模拟 Pred/Coeff 程序，并与 `ref_eval` 对比。
- **proto 在线层**：`reluars_online_complete.hpp`、`gelu_online_step_dcf.hpp` 等使用 PFSS 后端给出的 helper bits/系数 share，结合 BeaverMul64/BitRingOps 完成 MPC 逻辑，最终输出 masked y。
- **后端抽象**：
  - `proto/backend_clear.hpp`：确定性明文 share（party0 负载，party1=0），用于自测/harness。
  - `proto/myl7_fss_backend.hpp`：真实 myl7/fss 适配。若检测到 `<fss/dcf.h>`，使用 C 库 keygen/eval；payload 超过 `kLambda` 时自动按 `kLambda` 分块生成多把 DCF key 并拼回原始长度；key header 编码包含 in_bits / 分块数 / party 位 / payload_len。
  - `proto/sigma_fast_backend_ext.hpp`：packed compare / interval LUT 的 CPU 实现（party0 为真实掩码或 payload，party1=0），支持 `in_bits<=64`，带断言测试 `test_sigmafast`。

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

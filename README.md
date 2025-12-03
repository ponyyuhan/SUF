# SUF/PFSS Prototype (GPT-5 Codex Agent)

## 目录结构（核心部分）

```
CMakeLists.txt
include/
  core/               # 基础环与序列化
  mpc/                # Beaver 三元组、共享、通道抽象
  suf/                # SUF IR、谓词、布尔表达式、多项式
  compiler/           # SUF -> PFSS 编译描述与掩码旋转
  pfss/               # 抽象 PFSS 后端接口 + 明文/外部适配器
  gates/              # 门级 API + ReLU/ReluARS/GeLU 占位 SUF
  proto/              # 为 expand 流程新增的原型层（bits-in/bytes-out DCF）
    backend_clear.hpp         # 明文 DCF/区间 LUT 后端（测试用）
    beaver.hpp/secure_rand.hpp/common.hpp  # 原型层基础工具（统一复用 core/serialization pack/unpack）
    beaver_mul64.hpp/channel.hpp/bit_ring_ops.hpp # 在线乘法+布尔运算（批量 Beaver 默认一轮）
    tape.hpp                      # Vec/File Tape 写/读，标签化记录（u64/u64_vec/bytes/triple64）
    pfss_backend.hpp/_batch.hpp/pfss_interval_lut_ext.hpp/sigma_fast_backend_ext.hpp # 后端接口与扩展
    reluars_dealer.hpp / reluars_online_complete.hpp # ReluARS 生成+在线
    gelu_spline_dealer.hpp / gelu_online_step_dcf.hpp / gelu_online_interval_lut.hpp / gelu_batch_step_dcf.hpp # GeLU 样条生成与在线
    pfss_utils.hpp / pack_utils.hpp    # DCF eval 工具与 SoA 打包
    myl7_fss_backend.hpp               # myl7/fss 适配骨架（带 party-bit key 编码，仍为内存 stub）
src/demo/
  demo_relu.cpp / demo_gelu.cpp        # 最初的 SUF/门演示
  demo_proto_reluars.cpp / demo_proto_gelu.cpp  # proto 层演示
  demo_proto_endtoend.cpp              # 使用在线评估器的端到端（简化版）
  sim_harness.cpp                      # 全流程两方模拟 + 断言测试（使用明文后端）
expand*.md, initial.md, paper.md       # 需求与设计文档
```

## 关键逻辑说明

### SUF/编译/门（初始层）
- `include/suf/*` 定义 SUF IR：区间、多项式、布尔表达式、谓词（x<β，x mod 2^f < γ，MSB 等）。
- `include/compiler/*` 将 SUF 谓词/系数描述编译为 PFSS 可用的旋转区间描述，输出编译密钥结构 `CompiledSUFKeys`。
- `include/pfss/*` 定义 PFSS 抽象接口（ProgGen/Eval）及明文后端、外部适配器骨架。
- `include/gates/*` 提供门级评估 API（Horner + PFSS Eval + mask 加法），以及简单 ReLU SUF 占位。

### proto 层
面向 bits-in/bytes-out DCF 后端（如 myl7/fss），提供在线评估/打包/批处理骨架。
- 后端接口：`pfss_backend.hpp`（基类），`pfss_backend_batch.hpp`（批量 Eval），`pfss_interval_lut_ext.hpp`（区间 LUT 扩展），`sigma_fast_backend_ext.hpp`（未来 packed compare/interval LUT 的 stub）。
- 后端实现：
  - `backend_clear.hpp`：明文 DCF + 区间 LUT，输出 additive u64 share（key 编码包含 party bit，payload 随机拆分）。
  - `myl7_fss_backend.hpp`：myl7/fss 适配骨架（party-bit key），当前用内存映射存储，待替换为真实库调用。
- Beaver/通道/布尔运算：`beaver.hpp`（生成）、`beaver_mul64.hpp`（在线乘法，单个或批量）、`channel.hpp`（通用通道抽象）、`bit_ring_ops.hpp`（基于 u64 的 AND/OR/NOT/XOR/SEL/LUT）。
- Tape：`tape.hpp` 提供 Vec/File sink/source，记录格式含标签+len，支持 `u64`/`u64_vec`/`bytes`/`triple64(vec)`；`TapeReader`/`TapeWriter` 在在线/离线均可复用。
- ReluARS：
  - `reluars_dealer.hpp`：离线生成 r_in、r_hi、r_out 掩码 share，比较 DCF 密钥，以及 Beaver 三元组。
  - `reluars_online_complete.hpp`：在线 evaluator，从 DCF 导出 w/t/d，截断 q，ReLU，8 项 LUT 校正，输出 y 和 masked y。
- GeLU 样条：
  - `gelu_spline_dealer.hpp`：构建偏置域的分段系数 LUT（旋转+拆分 wrap），生成 DCF 密钥和 Beaver 三元组。
  - `gelu_online_step_dcf.hpp`：使用多次 DCF（step 形式）拉取系数向量，Horner 求 δ(x)，x⁺，输出 y。
  - `gelu_online_interval_lut.hpp`：单次区间 LUT（SIGMA 快路径目标）的评估器。
  - `gelu_batch_step_dcf.hpp`：批处理调度器骨架（SoA 按 cut 分打包，批量 Beaver 轮次）。
- 辅助：`pfss_utils.hpp`（DCF eval to u64/vec）、`pack_utils.hpp`（扁平化 key 为 [N][key_bytes]，支持 GPU/SoA 打包）。

### Tape 消费顺序（关键对齐保证）
- ReluARS（每实例每方）: `[wrap_flag][r_in_share][r_hi_share][r_out_share][k_hat_lt_r][k_hat_lt_r+2^63][k_low_lt_r_low][k_low_lt_r_low+1][triple64 vec]`
- GeLU step-DCF（每实例每方）: `[wrap_flag][r_in_share][r_out_share][k_hat_lt_r][k_hat_lt_r+2^63][base_coeff vec][num_cuts][per-cut dcf_key+delta_vec][triple64 vec]`

### 运行逻辑（数据流总览）
- Offline/dealer（proto 层）：根据公参生成掩码 share、DCFs（bits-in/bytes-out）、Beaver 三元组；可返回内存结构或按固定顺序写入 Tape（Vec/File）。
- Online/evaluator（proto 层）：给定 public `hatx`、通道、Tape/密钥，即可：
  1) 依序读 Tape（或直接用 in-memory key）重建本方密钥；
  2) 调用后端 DCF/Interval LUT 得到 helper bits/系数 share；
  3) 用批量 Beaver 做 AND/SEL/Horner，多轮量化在 u64 环中完成；
  4) 输出 y_share / masked y_share。
- 后端选择：`ClearBackend`（确定性 additive share，用于测试）、`Myl7FssBackend`（stub，待接真实 myl7/fss）、`SigmaFastBackend`（packed/interval LUT stub）。
- 打包/批处理：`pack_keys_flat`/`pack_cut_keys_by_cut` 生成 SoA 缓冲区；`eval_dcf_many_u64` 默认循环，可替换为高性能实现；BeaverMul64Batch 一轮完成批乘。
- Tape 使用：`TapeWriter`/`TapeReader` 统一标签+长度格式；Vec/File 源/汇可替换，不影响在线逻辑。

### 测试/演示
- `sim_harness.cpp`：自包含两方模拟，使用 `ClearBackend` + 在线评估器（ReluARS/GeLU），内存通道+线程，带断言计数，演示 key 打包。
  - 内置自测：批量 Beaver 乘法、Tape 写/读回环、Bit/LUT 逻辑正确性、`pack_keys_flat` + `eval_dcf_many` 对比逐次 eval。
  - ReluARS/GeLU 参考：ReluARS 使用截断+delta 的明文公式；GeLU 使用同一样条系数/切点（含 mask 旋转）明文计算。默认各 2000 轮随机输入，可通过 `RELU_ITERS`/`GELU_ITERS` 环境变量调整（如 10000）。
  - 可直接 `cmake --build build && ./build/sim_harness` 或 `g++ -O2 -std=c++20 src/demo/sim_harness.cpp -pthread -Iinclude -o sim_harness && ./sim_harness`。
- 其他 demo（`demo_proto_*`、`demo_proto_endtoend`）展示 proto 层接口。

## 状态与待办
- Milestone 1（runtime 基础/批处理）✅：单一 pack/unpack 实现、批量 Beaver 默认一轮、bit/LUT ops、pack_flat+eval_many 与逐点一致性测试。
- Milestone 2（Tape/顺序对齐）✅：Vec/File tape、明确标签格式、ReluARS/GeLU 写/读顺序固定、from-tape evaluator、harness 真实明文参考 2000+ 随机用例。
- 未完成/待接入（Milestone 3/4 尚未启动）：SUF IR 强化、mask-rewrite 引擎 (§3.3)、真实 Delta/LUT 数据、`Myl7FssBackend` 真实库对接、`SigmaFastBackend` packed compare/interval LUT。
  - 其他待办：精确的 Delta 表、真实 spline 系数/区间（当前为 toy/占位）。

## 构建与运行
- 现已全局使用 C++20：`cmake --build build`
- 运行模拟测试：`./build/sim_harness`（默认 2000 轮，可设置 `RELU_ITERS`/`GELU_ITERS`）
- 生成的可执行：
  - `demo_proto_*`, `demo_proto_endtoend`（原型演示）
  - `sim_harness`（断言式测试）

## 接入真实后端的建议
1) 在 `include/proto/myl7_fss_backend.hpp` 中替换 `gen_dcf`/`eval_dcf` 为 myl7/fss 的实际 API（bits-in/bytes-out），保持输出长度 8×words 供 u64 share 解码。
2) 若走 SIGMA 风格优化，实现 `SigmaFastBackend` 的 packed compare（CDPF）与 interval LUT（vector-payload），复用现有在线 evaluators，不改门级逻辑。
3) 将明确的 Delta 表、样条系数、区间参数填入 dealer，替换 toy 逻辑，并更新 harness 参考验证。

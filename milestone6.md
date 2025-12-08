## Milestone 9 — Add the missing LLM gates (minimum set)

### 9.0 总目标 & 统一约定（Fixed-point / 范围 / 误差口径）

**目标**：补齐 Transformer 推理常用的激活与归一化算子，并在 2PC（Clear/myl7/SigmaFast）下可测、可 bench。

**统一约定（建议写进 docs/spec_m9_llm_gates.md，并在代码里用同一套常量/模板）**

* **数值表示**：沿用你现有的 ring64 fixed-point 约定（若已有 `FracBits`/`Fixed` 类型就复用；否则在 core 里补一个轻量封装）。

    * 建议：`Q(F)`（例如 F=16 或你当前 GeLU/ReluARS 用的那套），约束所有新 gate 的 I/O scale 一致，避免多处 rescale。
* **clip 策略**（避免溢出/迭代发散）

    * `nExp(x)=exp(-x)`：输入强制 `x ∈ [0, XMAX]`（建议 `XMAX=16`）。
    * `reciprocal(x)`：输入强制 `x ∈ [1, NMAX]`（Softmax sum 至少为 1）。
    * `rsqrt(x)`：输入强制 `x ∈ [EPS, VMAX]`（EPS 用可表示的固定点常数）。
* **正确性口径**：

    * “2PC correctness test” 对齐 **cleartext ref（同一近似算法）**，做到 **bit-exact**（最稳）。
    * 另加一个“accuracy smoke test”（可选但强烈建议）对齐 `double` 真值，允许误差（用于防止近似表/系数搞错）。

---

## 9.1 新增：通用“分段多项式/样条”评估骨架（给 SiLU/exp/recip/rsqrt 共用）

### 代码落点

* `include/gates/piecewise_poly.hpp`

    * Piecewise polynomial spec + Horner evaluator
* `include/gates/spline_eval.hpp`（如果已存在 gelu spline 的实现就复用/抽出来）
* `include/gates/tables/`

    * `silu_spline_table.hpp`
    * `nexp_piecewise_table.hpp`
    * `recip_piecewise_affine_init.hpp`（若用 NR：存分段 affine 初值）
    * `rsqrt_piecewise_affine_init.hpp`
* （可选）`tools/gen_tables/*.py`

    * 离线生成系数表，输出 C++ header（把生成脚本和生成结果一起提交，防以后手改）

### 关键接口（你可以按现有风格微调命名）

* `PiecewisePolySpec`

    * `std::span<const int64_t> breaks`（分段边界，fixed-point）
    * `std::span<const CoeffPack> coeffs`（每段的系数 pack，Horner 顺序）
    * `int frac_bits_in, frac_bits_out`
    * `clip_lo/clip_hi`
* `eval_piecewise_poly(ctx, xs, spec, out)`

    * 用你现有 `interval LUT selection network` 产 **one-hot interval masks**（或等价 index）
    * `coeff_selected = Σ mask_i * coeff_i`（系数是 public，mask 是 secret：乘常数是线性操作，本地就能做）
    * Horner 时用你现有 batched Beaver mul + rescale

> 这一步的意义：后面 4 个标量 gate 都变成“选系数 + Horner + 少量迭代”，不会各写各的一坨。

---

## 9.2 Gate #1：**SiLU（spline）**

### 数学形式

* `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`
* 实现用 **直接逼近 SiLU 的样条**（比“先 sigmoid 再乘”更省）

### 近似建议（最小可用）

* domain：`x ∈ [-8, 8]`
* clamp：

    * `x <= -8 => 0`
    * `x >= 8 => x`（因为 sigmoid→1）
* 中间用 **分段三次样条**（比如 16 或 32 段；段数可通过 bench 调）
* 输出保持同一 fixed-point scale

### 代码落点

* `include/gates/silu_spline_gate.hpp`
* `include/gates/tables/silu_spline_table.hpp`

### DOD 对应

* cleartext ref：`ref_silu_spline_fixed(x)`（同 table 同 rounding）
* 2PC test：随机 x（覆盖负数、接近拐点、极值）
* bench：批量 N（比如 1e5/1e6）测吞吐

---

## 9.3 Gate #2：**exp approx（推荐实现 nExp：exp(-x)）**

Transformer 里 softmax 稳定版基本都会走 `exp(x-max)`，等价 `nExp(max-x)`，更好控范围。

### 近似建议（最小可用）

* 定义：`nExp(t) = exp(-t)`，输入 `t >= 0`
* domain：`t ∈ [0, 16]`（Softmax 差值常见足够）
* clamp：`t < 0 => 0`（数值上当 0），`t > 16 => 16`
* 分段多项式（每段 1.0 或 0.5 宽度；deg=3~5）

    * 段宽大：更快；段宽小：更准（先用 16 段 deg=4 起步）

### 代码落点

* `include/gates/nexp_gate.hpp`
* `include/gates/tables/nexp_piecewise_table.hpp`

### DOD 对应

* cleartext ref：`ref_nexp_piecewise_fixed(t)`
* 2PC test：随机 t（重点测 t≈0、t≈16、段边界）
* bench：N 大批量（复用你 SigmaFast packed/SoA 路径）

---

## 9.4 Gate #3：**reciprocal（softmax 归一化）**

Softmax 的 `sum(exp)` 至少为 1（包含 max 那一项 exp(0)=1），所以 reciprocal 的输入范围很好控。

### 推荐实现（最小可用 + 性能友好）

**分段 affine 初值 + 1~2 次 Newton-Raphson**

* 目标：`y ≈ 1/x`
* NR：`y <- y * (2 - x*y)`
* 步骤：

    1. interval masks（用 breakpoints：`[2,4,8,...,NMAX]`）定位 `x ∈ [2^k,2^{k+1})`
    2. 分段选 affine 初值：`y0 = a_k - b_k * x`
    3. 做 1 次 NR（通常够用），可配置 2 次（更准）

### 参数建议

* `NMAX`：按你的模型上限定（例如 head dim 或序列长度上界），默认先给 1024/4096
* clip：`x ∈ [1, NMAX]`

### 代码落点

* `include/gates/reciprocal_gate.hpp`
* `include/gates/tables/recip_piecewise_affine_init.hpp`

### DOD 对应

* cleartext ref：`ref_recip_nr_fixed(x)`（同样的 affine+NR 流程）
* 2PC test：随机 x ∈ [1,NMAX]（多测靠近 1 和靠近 NMAX）
* bench：吞吐 + 可选把 NR iter 数做成参数

---

## 9.5 Gate #4：**rsqrt（LayerNorm）**

LayerNorm 的 `variance+eps` 通常在 1 附近（训练分布决定），所以只要把 domain 控在一个合理区间就能很实用。

### 推荐实现（最小可用）

**分段 affine 初值 + 1 次 rsqrt Newton**

* 目标：`y ≈ 1/sqrt(x)`
* 迭代：`y <- y * (1.5 - 0.5 * x * y^2)`
* domain：建议先 clip 到 `[EPS, 16]` 或 `[EPS, 8]`（由 bench/模型决定）
* EPS：用 fixed-point 可表示的常数（比如 `2^-10` 量级，避免因为 1e-5 在小 frac_bits 下变 0）

### 代码落点

* `include/gates/rsqrt_gate.hpp`
* `include/gates/tables/rsqrt_piecewise_affine_init.hpp`

### DOD 对应

* cleartext ref：`ref_rsqrt_nr_fixed(x)`
* 2PC test：随机 x（覆盖 EPS 附近、1 附近、VMAX 附近）
* bench：N 大批量

---

## 9.6 Vector Block #1：**SoftmaxBlock**

### 计算图（明确要求的 max/exp/sum/recip/mul）

1. `m = max(x[0..L-1])`
2. `t_i = m - x_i`（确保非负；必要时 clip 到 [0,16]）
3. `e_i = nExp(t_i)`
4. `s = Σ e_i`
5. `inv = reciprocal(s)`
6. `y_i = e_i * inv`

### 实现关键点（结合你现有资产）

* **max reduction**

    * 最小实现：线性扫描 `cur=max(cur, x_i)`

        * compare：复用你现有 predicate/bitops 能力（优先走 SigmaFast packed compare）
        * select：`cur + bit*(x_i-cur)`（bit 为 0/1 的 ring share）
    * 性能进阶：树形 reduce（更适合 batch）
* exp/sum/recip/mul 全部走你上面实现的标量 gates（batched）

### 代码落点

* `include/gates/softmax_block.hpp`
* demo/test：`src/demo/test_softmax_block.cpp`
* bench：`src/bench/bench_softmax_block.cpp`（或合并 `bench_llm_blocks.cpp`）

### DOD 对应

* cleartext ref：`ref_softmax_block_fixed(vec)`（同 max + same nExp/recip）
* 2PC test：随机向量（L=32/128/768 至少测 2-3 个典型长度）
* bench：batch 维度 + L 维度可调（输出 token/s 或 elem/s）

---

## 9.7 Vector Block #2：**LayerNorm（mean/var/rsqrt/affine）**

### 计算图（标准 LN）

给定 `x[0..L-1]`，可选 `gamma/beta`：

1. `mu = (Σ x_i) * invL`
2. `d_i = x_i - mu`
3. `var = (Σ d_i^2) * invL`
4. `r = rsqrt(var + eps)`
5. `z_i = d_i * r`
6. `y_i = z_i * gamma_i + beta_i`（可选；gamma/beta 可做 public 或 secret-share 两种路径）

### 实现关键点

* `invL`、`eps` 都是 public 常量（fixed-point），乘常数是线性本地操作
* `d_i^2`、`z_i*gamma_i` 等乘法用 batched Beaver
* `rsqrt` 调用上面的 rsqrt gate

### 代码落点

* `include/gates/layernorm_block.hpp`
* demo/test：`src/demo/test_layernorm_block.cpp`
* bench：`src/bench/bench_layernorm_block.cpp`

### DOD 对应

* cleartext ref：`ref_layernorm_fixed(x,gamma,beta,eps)`
* 2PC test：随机向量 + 随机 gamma/beta（至少一组测试把 gamma/beta 设为 1/0 做 sanity）
* bench：L=768/1024 等典型维度、batch N 可调

---

## 9.8 Compiler / Program Descriptor 接入（让这些 gate 可被“编排”）

即使你暂时不把它们塞进 SUF→PFSS 自动识别，也建议把它们作为 **PFSS program 的内建 gate kind**，这样 demo/bench 可以用统一的 composite pipeline 跑。

### 代码落点（最小改动）

* `include/compiler/pfss_program_desc.hpp`

    * `enum class GateKind` 增加：

        * `SiLUSpline`, `NExp`, `Reciprocal`, `Rsqrt`, `SoftmaxBlock`, `LayerNormBlock`
    * 增加对应 `GateParams`（至少需要 frac_bits、clip、table_id、L、eps、nr_iters）
* `include/compiler/compiled_gate_metadata.hpp`（或你现有那个元数据文件）

    * 对 block gates 加 `PortLayout` 命名约定（例如 `x[0..L-1]`, `y[0..L-1]`, `gamma[]`, `beta[]`）
* `src/compiler/*`：

    * 最小：支持把这些 gate 作为 program descriptor 的节点执行（不要求 SUF pattern 识别）

---

## 9.9 Tests（2PC correctness）& Demos（clear ref）

### 建议新增 demo/test 文件

* `src/demo/test_silu_gate.cpp`
* `src/demo/test_nexp_gate.cpp`
* `src/demo/test_recip_gate.cpp`
* `src/demo/test_rsqrt_gate.cpp`
* `src/demo/test_softmax_block.cpp`
* `src/demo/test_layernorm_block.cpp`

### 每个测试都做三件事

1. 随机输入生成（固定 seed，覆盖边界/段边界/clip 边界）
2. cleartext ref（同近似算法）得到期望输出
3. 2PC 跑一遍（建议至少 SigmaFast；最好 Clear/myl7/SigmaFast 都跑一遍），重构 output，**bit-exact** 或容忍 1-2 LSB（看你内部 rounding 是否完全一致）

---

## 9.10 Bench（batched）

### 建议新增

* `src/bench/bench_llm_gates.cpp`（统一入口，命令行选择 gate）

    * `--gate=silu|nexp|recip|rsqrt|softmax|layernorm`
    * `--N=...`（batch size）
    * `--L=...`（vector len for blocks）
    * `--iters=...`（NR 迭代次数、或重复次数）
* 输出指标：

    * `elem/s`（标量）
    * `vector/s` 或 `token/s`（block）
    * 以及可选：通信量（如果你 bench 框架已有统计）

---

# Milestone 9 — Definition of Done（逐条可验收）

对 **每个 gate / block**（SiLU、nExp、recip、rsqrt、SoftmaxBlock、LayerNorm）都满足：

1. **cleartext ref**：对应 `ref_*`（同近似/同 fixed rounding）
2. **2PC correctness test**：随机输入通过（建议 SigmaFast + 至少一个对照 backend）
3. **batched benchmark**：能跑、能输出吞吐（并在 docs 里记录典型参数下结果）

---

如果你愿意，我也可以直接按你现有命名习惯，把 **每个新增 header 的类/函数签名**、以及 **Softmax/LayerNorm 的 PortLayout 命名方案**写成更接近“可复制进代码”的接口草案（不需要你再补问题，我会按你当前布局默认最兼容的方案来写）。

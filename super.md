## 0. Design snapshot / constraints

Given what you wrote, I’ll assume:

* `PfssSuperBatch::view(PfssHandle)` returns a `PfssResultView` with:

    * `const uint64_t* arith;` (host)
    * `const uint8_t* bools;`
    * `size_t arith_words;`
    * `size_t r;`  // arith words per element
    * `size_t ell;` // predicate bits per element
* CUDA backend already keeps PFSS **outputs** on device; at the moment:

    * It copies them back to host so `arith`/`bools` point to host.
* Postproc (trunc, Horner, Recip NR, MulRowBroadcast) is entirely CPU-side through `PostProcHook::run_batch` + `BeaverMul64`.

Goal:

* **Keep PFSS outputs stored on device** (and optionally still mirror small slices on host for debug).
* Add **CUDA kernels** for:

    * Faithful/GapARS trunc postproc (mask removal, sign, shift).
    * Horner evaluation of cubic SUFs (SiLU/nExp/Recip).
    * Beaver-style mul/add for generic MulTask/MulRowBroadcast and NR iterations.
* Extend **PfssResultView** + **PhaseTasks** to branch into device path when device outputs are available, while preserving existing CPU behavior.

---

## 1. Add device views to PFSS results

### 1.1 Extend `PfssResultView`

In your PFSS backend headers (e.g. `pfss_backend.hpp`), extend the view:

```cpp
struct PfssResultView {
    // Existing host pointers:
    const uint64_t* arith = nullptr;     // host
    const uint8_t* bools   = nullptr;    // host
    size_t arith_words = 0;
    size_t r = 0;   // arith words per element (r_out)
    size_t ell = 0; // predicate bits per element

    // NEW: device pointers (nullptr for non-CUDA backends)
    const uint64_t* d_arith = nullptr;
    const uint8_t* d_bools  = nullptr;
    size_t d_arith_words = 0;    // may equal arith_words
    // Optionally, a stream for this result:
    void* device_stream = nullptr; // reinterpret_cast<cudaStream_t>(...)
    bool on_device = false;        // true if CUDA backend owns device buffers
};
```

For **CPU backends**: set `on_device=false`, `d_* = nullptr`.

For the **CUDA backend**:

* Allocate a device buffer for PFSS output (arith + bools) and fill `d_arith`, `d_bools`, `d_arith_words`.
* Do **not** eagerly copy back to host; copy only if a CPU consumer asks.

### 1.2 Composite output hook in CUDA backend

Add a “composite” call in the CUDA backend to perform PFSS eval into device buffers:

```cpp
class CudaPfssBackend : public PfssBackend {
 public:
  PfssResultView eval_batch_device(const CompiledProgram& prog,
                                   const uint64_t* d_hatx,
                                   size_t n_elems,
                                   cudaStream_t stream);
  // existing eval_batch_host(...) unchanged
};
```

`eval_batch_device`:

* Reads gates, seeds, descriptors.
* Generates device arith/bools into `d_arith` / `d_bools`.
* Returns `PfssResultView` with `on_device = true`.

`PfssSuperBatch::flush_eval` should detect a CUDA backend and call `eval_batch_device` when `hatx` is already on device (or copy it as needed).

---

## 2. GPU kernels: postproc + Horner + Beaver

### 2.1 Device Beaver triple pool

You don’t want to copy triples for every task.

Define a device pool (e.g. in `proto/device_beaver.hpp`):

```cpp
struct DeviceBeaverTriplePool {
    uint64_t* d_a = nullptr;
    uint64_t* d_b = nullptr;
    uint64_t* d_c = nullptr;
    size_t capacity = 0;
};

struct DevicePartyContext {
    int party;
    DeviceBeaverTriplePool triples;
    // possibly pointer to per-layer offset, etc.
};
```

CUDA backend fills this pool once (per layer/batch):

* Copy `CompositePartyKey::triples` to device linearly.
* Let tasks index it by a base index (`triple_base_offset`).

### 2.2 Generic Beaver mul kernel

Kernel interface (in `src/runtime/cuda_beaver_kernels.cu`):

```cpp
__global__ void beaver_mul_kernel(
    int party,
    const uint64_t* __restrict__ d_x,   // share of x
    const uint64_t* __restrict__ d_y,   // share of y
    const uint64_t* __restrict__ d_a,   // triple a
    const uint64_t* __restrict__ d_b,   // triple b
    const uint64_t* __restrict__ d_c,   // triple c
    const uint64_t* __restrict__ d_d_open, // opened d = x - a
    const uint64_t* __restrict__ d_e_open, // opened e = y - b
    uint64_t* __restrict__ d_out,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t x = d_x[idx];
    uint64_t y = d_y[idx];
    uint64_t a = d_a[idx];
    uint64_t b = d_b[idx];
    uint64_t c = d_c[idx];
    uint64_t d_open = d_d_open[idx];
    uint64_t e_open = d_e_open[idx];

    uint64_t z = c;
    z = proto::add_mod(z, proto::mul_mod(d_open, b));
    z = proto::add_mod(z, proto::mul_mod(e_open, a));
    if (party == 0) {
        z = proto::add_mod(z, proto::mul_mod(d_open, e_open));
    }
    d_out[idx] = z;
}
```

Note:

* Opens (`d_open`, `e_open`) are still done on host via `OpenCollector`.
* After open, host copies `open_d/e` to `d_d_open/d_e_open` and launches kernel.
* This already removes O(n) multiplies from CPU.

### 2.3 Trunc postproc kernel (GapARS + faithful)

You already have C++ code for trunc postproc. Port the per-element logic to CUDA; for each element:

* Read `x_share` OR `PFSS output` words slice.
* Apply sign/wrap logic and shift. Use compile-time specialization for `GapARS` vs `Faithful`.

Skeleton (you’ll fill exact arithmetic):

```cpp
enum TruncKind : uint8_t { TK_Faithful, TK_GapARS };

__global__ void trunc_postproc_kernel(
    TruncKind kind,
    int frac_bits_src,     // e.g., 2f or 3f
    int frac_bits_dst,     // target f
    int signed_value,
    const uint64_t* __restrict__ d_in,   // PFSS arith outputs (Q?f)
    uint64_t* __restrict__ d_out,        // final truncated share (Qf)
    const uint8_t* __restrict__ d_wrap,  // optional wrap/sign mask
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t v = d_in[idx];  // Q?f

    // Example: faithful truncation = floor(v / 2^(frac_bits_src - frac_bits_dst))
    int shift = frac_bits_src - frac_bits_dst;
    // apply wrap/sign logic depending on kind and signed_value
    // ... your exact GapARS / faithful formulas here ...

    d_out[idx] = truncated_value;
}
```

For simplification you can **first** handle the “no wrap” case (already used when GapCert ensures no wrap), then extend to full GapARS.

### 2.4 Horner cubic kernel

Horner for cubic:

> y = (((c3 * x) + c2) * x + c1) * x + c0

Where c_i are PFSS coeff outputs, x is share in Q^f (or x̂ if polynomial is in masked domain). For your current CPU CubicPolyTask, you do 3 Beaver mul + 2 Trunc.

On GPU:

* Keep the **Horner iteration itself on device** using `beaver_mul_kernel` and `trunc_postproc_kernel`.
* The high-level sequencing remains in host (task state machine), but each “MulTask” / “TruncTask” step uses GPU kernels.

Pseudo-kernel where we assume all inputs are secret shares and we already have triples + open d/e on device:

```cpp
__global__ void horner_cubic_step_kernel(
    const uint64_t* __restrict__ d_x,
    const uint64_t* __restrict__ d_c0,
    const uint64_t* __restrict__ d_c1,
    const uint64_t* __restrict__ d_c2,
    const uint64_t* __restrict__ d_c3,
    const uint64_t* __restrict__ d_trunc1_in, // p (Q3f)
    const uint64_t* __restrict__ d_trunc2_in, // r (Q3f)
    uint64_t* __restrict__ d_y_out,           // Qf
    int frac_bits,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Using pre-truncated p2, r2 etc. from separate Trunc kernels
    uint64_t c0 = d_c0[idx];
    uint64_t c1 = d_c1[idx];
    uint64_t c2 = d_c2[idx];
    uint64_t c3 = d_c3[idx];
    uint64_t x  = d_x[idx];

    // Here you can just do:
    // m1 = c3*x (Beaver mul already done)
    // p2, r2 already truncated
    // final: y = r2 + c0
    // (the other steps have been done separate)
    uint64_t y = proto::add_mod(d_trunc2_in[idx], c0);
    d_y_out[idx] = y;
}
```

In practice you’ll have:

* `beaver_mul_kernel` for each cubic mul stage.
* `trunc_postproc_kernel` for each trunc stage.
* A final simple “+c0” kernel.

This matches your current CubicPolyTask structure, just swapping CPU loops+BeaverMul64+TruncTask to GPU kernels.

---

## 3. Wiring into existing tasks

### 3.1 Extend `PhaseResources` with optional device context

In `phase_executor.hpp`:

```cpp
struct DeviceResources {
    void* device_stream = nullptr;     // cudaStream_t
    DeviceBeaverTriplePool* triples = nullptr;
    // possibly device allocator, etc.
};

struct PhaseResources {
  int party = 0;
  proto::PfssBackendBatch* pfss_backend = nullptr;
  proto::IChannel* pfss_chan = nullptr;
  net::Chan* net_chan = nullptr;
  PfssSuperBatch* pfss_coeff = nullptr;
  PfssSuperBatch* pfss_trunc = nullptr;
  OpenCollector* opens = nullptr;

  // NEW:
  DeviceResources* device = nullptr;  // nullptr = CPU-only
};
```

In the CUDA path (e.g. `nn/transformer_layer.cpp`), when constructing `PhaseResources R;`:

* If you have a CUDA backend, build a `DeviceResources` instance with `cudaStream_t` and `DeviceBeaverTriplePool` and set `R.device = &dev_res`.

### 3.2 Device branch in `TruncTask::step`

Locate the part where you currently do:

```cpp
auto v = R.pfss_trunc->view(PfssHandle{token_});
size_t elems = in_.size();
std::vector<uint64_t> hook_out(elems * v.r, 0);
proto::BeaverMul64 mul{R.party, *R.pfss_chan, *triples};
hook_->run_batch(... v.arith, v.r, v.bools, v.ell, elems, hook_out.data());
for (size_t i = 0; i < elems; ++i) out_[i] = hook_out[i * v.r];
```

Refactor to:

```cpp
auto v = R.pfss_trunc->view(PfssHandle{token_});
size_t elems = in_.size();

// Device path:
if (R.device && v.on_device && v.d_arith && R.device->triples) {
    // 1) arrange device views:
    const uint64_t* d_pfss = v.d_arith; // PFSS raw outputs per element
    uint64_t* d_out = /* device buffer for truncated outputs, e.g. from a pool or cudaMallocAsync */

    // 2) launch trunc_postproc_kernel with correct kind/shift/sign:
    dim3 block(256);
    dim3 grid((elems + block.x - 1) / block.x);
    trunc_postproc_kernel<<<grid, block, 0,
                            (cudaStream_t)R.device->device_stream>>>(
        bundle_->kind, bundle_->src_frac_bits, bundle_->dst_frac_bits,
        /*signed_value=*/1,
        d_pfss, d_out,
        /*d_wrap=*/nullptr,
        elems
    );

    // 3) optionally copy back to host (only if subsequent code needs host):
    if (need_host_out_) {
        cudaMemcpyAsync(out_.data(), d_out, elems * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost,
                        (cudaStream_t)R.device->device_stream);
        cudaStreamSynchronize((cudaStream_t)R.device->device_stream);
    } else {
        // store device pointer somewhere in ctx for later GPU consumers
        ctx->store_trunc_device_output(d_out, elems);
    }

    st_ = St::Done;
    return detail::Need::None;
}

// Fallback CPU path:
std::vector<uint64_t> hook_out(elems * v.r, 0);
proto::BeaverMul64 mul{R.party, *R.pfss_chan, *triples};
hook_->run_batch(... v.arith, v.r, v.bools, v.ell, elems, hook_out.data());
for (size_t i = 0; i < elems; ++i) out_[i] = hook_out[i * v.r];
st_ = St::Done;
return detail::Need::None;
```

The `need_host_out_` flag can be:

* `true` for now to keep behavior identical;
* later, for device-only pipelines (softmax/LN on GPU), you set it `false` and thread a `DeviceTensor` through `LayerContext`.

### 3.3 Device branch in `RecipTask` & `MulRowBroadcastTask`

Exactly the same pattern:

* Where you call `MulTask` (CPU) → replace with:

    * compute d,e opens on host as today;
    * copy d,e to device;
    * launch `beaver_mul_kernel` with device triples.
* Where you call `TruncTask` (CPU) inside RecipTask → call device trunc kernel if available.
* For MulRowBroadcast:

  You already compute D (matrix) and e (vector) and open them via OpenCollector. After open:

    * Copy `opened_D` and `opened_e` to device: `d_d_open`, `d_e_open`.
    * Launch `beaver_mul_kernel` row-wise with x=mat, y=vec (repeated), triples from device pool.

---

## 4. End-to-end GPU path for softmax/LN

Once the building blocks above exist and are tested:

1. In **softmax task**:

    * Keep scores, exp outputs, sum, recip, prob, etc., as **device buffers** for the CUDA case.
    * Use:

        * PFSS on device → `PfssResultView.on_device=true`.
        * `CubicPolyTask` steps calling `beaver_mul_kernel` + `trunc_postproc_kernel` and writing to device.
        * `RecipTask` similarly.

2. In **LayerNormTask**:

    * Means/vars can be computed via device reductions (you can either use your own simple reduction kernel or call cuBLAS/cuDNN later).
    * Rsqrt via device Recip/RsqrtTask.

3. Only at the **very end of transformer_layer**, if the caller is CPU-only, copy the final `Y_share` back to host; otherwise, keep it on device for the next layer.

---

## 5. Testing strategy

You absolutely want CUDA-gated tests before turning this on in real runs.

### 5.1 Micro-tests per primitive

1. `test_cuda_trunc_postproc`:

    * Generate random Q3f values on host.
    * Run CPU trunc (GapARS + faithful) and device trunc kernel.
    * Compare outputs bit-exact for 1k–10k elements.

2. `test_cuda_beaver_mul`:

    * Generate random x/y and triples; run full Beaver protocol CPU-side to get the expected z share.
    * For d/e, reuse the same opened values, but call the device kernel to compute z.
    * Compare z_cpu vs z_gpu.

3. `test_cuda_cubic_poly`:

    * Use your CPU `CubicPolyTask` to compute SiLU/nExp for a small tensor.
    * Using the same PFSS coeffs and triples, run the device version (Hornder + trunc on GPU).
    * Compare outputs.

### 5.2 Softmax/LN smoke

`test_softmax_ln_gpu_smoke`:

* Small B,H,T (e.g. 1×2×8).
* Run CPU pipeline and GPU pipeline end-to-end (attention+softmax+LN).
* Compare outputs (allow for optional small differences if you ever switch to float intermediate; for now, you can keep them bit-exact).

### 5.3 Benchmarks

Extend `bench_softmax_norm`:

* Add flags:

    * `--mode=cpu|gpu|gpu_device_only`
* In `gpu_device_only`:

    * Don’t copy intermediate outputs back to host; only copy final Y (or even skip and just time kernels).
* Print:

    * PFSS time vs device postproc/Horner/Beaver time vs any D2H time.

That will tell you whether the device math is buying you anything for your actual shapes.

---

## 6. Recommended implementation order

If you follow this order，你的每一步都可以独立合进去，不会把整个系统弄崩：

1. **PfssResultView + device pointers**

    * Make CUDA backend fill `d_arith/d_bools/on_device` but still copy back to host as today.
    * No behavior changes; just more metadata.

2. **Device Beaver pool + beaver_mul_kernel**

    * Add pool + kernel + micro-test (CPU vs GPU Beaver mul).
    * Not yet used by tasks.

3. **trunc_postproc_kernel + test_cuda_trunc_postproc**

    * Implement faithful first; GapARS as second commit.
    * Still not wired into tasks, only tested alone.

4. **TruncTask GPU branch**

    * If `v.on_device && R.device`, call device trunc kernel; else fallback CPU.
    * Add `test_trunc_task_cuda` comparing CPU vs GPU.

5. **MulRowBroadcast + RecipTask GPU branch**

    * Use beaver_mul_kernel + trunc kernel.
    * Add corresponding CUDA tests.

6. **Wire softmax/LN tasks for GPU**

    * Keep data on device through softmax/LN; only D2H at layer boundary.
    * Extend `test_softmax_gpu_smoke` / `test_layernorm_task` with CUDA path.

7. **Overlap / stream tuning**

    * Once correctness is solid, start playing with stream concurrency and measuring speedups.

---

如果你愿意，下一步你可以把当前 `PfssResultView` / `TruncTask` 的具体定义贴（或已经贴过的一版保持不变），我可以帮你写一份更接近真实代码风格的 **C++ diff 样例**：包括 `PfssResultView` 新字段、`TruncTask` 中 GPU 分支的完整代码骨架，以及一个最小的 `test_cuda_trunc_postproc` 测试结构。

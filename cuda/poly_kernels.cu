#include <cstddef>
#include <cstdint>

// Sketch only: each thread evaluates one polynomial output for one element.
extern "C" __global__
void horner_eval_u64(const uint64_t* x_share,
                     const uint64_t* coeffs,  // [count*(d+1)]
                     uint64_t* out_share,
                     int d, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint64_t* c = coeffs + i * static_cast<size_t>(d + 1);
  uint64_t acc = c[d];
  uint64_t x = x_share[i];
  for (int k = d - 1; k >= 0; --k) acc = acc * x + c[k];
  out_share[i] = acc;
}

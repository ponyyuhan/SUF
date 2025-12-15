#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/pfss_superbatch.hpp"

static bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

int main() {
  runtime::PfssSuperBatch b;

  // Byte-limit enforcement should be independent of the word limit.
  runtime::PfssSuperBatch::Limits lim;
  lim.max_pending_jobs = 16;
  lim.max_pending_hatx_words = 1024;
  lim.max_pending_hatx_bytes = 16;  // 2 u64 words
  lim.max_flushes = 16;
  b.set_limits(lim);

  runtime::PreparedCompositeJob j0;
  j0.hatx_public = {1, 2};  // 16 bytes
  b.enqueue_composite(j0);

  runtime::PreparedCompositeJob j1;
  j1.hatx_public = {3};  // would push pending bytes to 24
  assert(throws_with([&] { b.enqueue_composite(j1); }, "pending hatx byte limit exceeded"));

  return 0;
}

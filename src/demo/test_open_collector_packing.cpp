#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "runtime/open_collector.hpp"

namespace {

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
    size_t sent0to1 = 0;
    size_t sent1to0 = 0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  LocalChan() = default;
  LocalChan(Shared* s, bool p) : sh(s), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q0to1 : sh->q1to0;
    q.push(v);
    if (is0) {
      sh->sent0to1 += 1;
    } else {
      sh->sent1to0 += 1;
    }
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

}  // namespace

static void run_case(bool enable_pack, size_t expected_words_each_direction) {
  if (enable_pack) {
    setenv("SUF_OPEN_PACK_EFFBITS", "1", 1);
    setenv("SUF_OPEN_PACK_MAX_BITS", "12", 1);
  } else {
    unsetenv("SUF_OPEN_PACK_EFFBITS");
    unsetenv("SUF_OPEN_PACK_MAX_BITS");
  }

  runtime::OpenCollector opens0, opens1;
  LocalChan::Shared sh;
  LocalChan ch0(&sh, true), ch1(&sh, false);

  // Request 1: values fit in <= 10 bits; when packing is enabled, this should
  // take the packed path.
  std::vector<uint64_t> a0 = {1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 0, 5, 9, 17, 33, 65, 129};
  std::vector<uint64_t> a1 = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 7, 0, 3, 1, 2, 3, 4};

  // Request 2: includes a high-bit value; must use the uncompressed path.
  std::vector<uint64_t> b0 = {0, 1, 2, 3, (uint64_t(1) << 63)};
  std::vector<uint64_t> b1 = {5, 6, 7, 8, 9};

  auto h0a = opens0.enqueue(a0);
  auto h1a = opens1.enqueue(a1);
  auto h0b = opens0.enqueue(b0);
  auto h1b = opens1.enqueue(b1);

  std::exception_ptr exc;
  std::thread t1([&] {
    try {
      opens1.flush(/*party=*/1, ch1);
    } catch (...) {
      exc = std::current_exception();
    }
  });
  try {
    opens0.flush(/*party=*/0, ch0);
  } catch (...) {
    exc = std::current_exception();
  }
  t1.join();
  if (exc) std::rethrow_exception(exc);

  auto v0a = opens0.view(h0a);
  auto v1a = opens1.view(h1a);
  assert(v0a.size() == a0.size());
  assert(v1a.size() == a1.size());
  for (size_t i = 0; i < a0.size(); ++i) {
    uint64_t expect = a0[i] + a1[i];
    assert(v0a[i] == static_cast<int64_t>(expect));
    assert(v1a[i] == static_cast<int64_t>(expect));
  }

  auto v0b = opens0.view(h0b);
  auto v1b = opens1.view(h1b);
  assert(v0b.size() == b0.size());
  assert(v1b.size() == b1.size());
  for (size_t i = 0; i < b0.size(); ++i) {
    uint64_t expect = b0[i] + b1[i];
    assert(v0b[i] == static_cast<int64_t>(expect));
    assert(v1b[i] == static_cast<int64_t>(expect));
  }

  assert(sh.sent0to1 == expected_words_each_direction);
  assert(sh.sent1to0 == expected_words_each_direction);
}

int main() {
  // Packing enabled: flush handshake (1 word) + req1 header (1) + req1 packed (3)
  // + req2 header (1) + req2 raw (5) = 11 words in each direction.
  run_case(/*enable_pack=*/true, /*expected_words_each_direction=*/11);

  // Packing disabled: flush handshake (1 word) + raw request words (17 + 5) = 23.
  run_case(/*enable_pack=*/false, /*expected_words_each_direction=*/23);
  return 0;
}

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>

#include "mpc/net.hpp"

namespace mpc {
namespace net {

// Simple in-process two-party channel for tests.
class LocalChan : public ::net::Chan {
 public:
  struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1;
    std::queue<uint64_t> q1to0;
  };

  LocalChan(Shared* s, bool is_party0) : sh_(s), is0_(is_party0) {
    if (!sh_) throw std::runtime_error("LocalChan: shared state is null");
  }

  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh_->mu);
    auto& q = is0_ ? sh_->q0to1 : sh_->q1to0;
    q.push(v);
    sh_->cv.notify_all();
  }

  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh_->mu);
    auto& q = is0_ ? sh_->q1to0 : sh_->q0to1;
    sh_->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }

 private:
  Shared* sh_;
  bool is0_;
};

}  // namespace net
}  // namespace mpc

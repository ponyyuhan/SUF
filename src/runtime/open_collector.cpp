#include "runtime/open_collector.hpp"

#include <chrono>
#include <stdexcept>

namespace runtime {

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff) {
  size_t new_pending = pending_words_ + diff.size();
  if (limits_.max_pending_words > 0 && new_pending > limits_.max_pending_words) {
    throw std::runtime_error("OpenCollector: pending open budget exceeded");
  }
  auto slot = std::make_shared<OpenSlot>();
  slot->n = diff.size();
  slot->opened.resize(diff.size(), 0);
  OpenHandle h;
  h.slot = slot;
  h.offset = 0;
  h.len = diff.size();
  Request r;
  r.diff = diff;
  r.slot = slot;
  r.offset = h.offset;
  r.len = h.len;
  requests_.push_back(std::move(r));
  pending_words_ = new_pending;
  if (pending_words_ > stats_.max_pending_words) {
    stats_.max_pending_words = pending_words_;
  }
  return h;
}

void OpenCollector::flush(int party, net::Chan& ch) {
  if (requests_.empty()) return;
  auto t0 = std::chrono::steady_clock::now();
  size_t total_words = 0;
  for (const auto& req : requests_) {
    if (!req.slot) {
      throw std::runtime_error("OpenCollector: missing result slot");
    }
    if (req.offset + req.len > req.slot->opened.size()) {
      throw std::runtime_error("OpenCollector: request out of range");
    }
    if (party == 0) {
      for (auto v : req.diff) ch.send_u64(v);
      for (size_t i = 0; i < req.len; ++i) {
        req.slot->opened[req.offset + i] = static_cast<int64_t>(req.diff[i] + ch.recv_u64());
      }
    } else {
      for (size_t i = 0; i < req.len; ++i) {
        req.slot->opened[req.offset + i] = static_cast<int64_t>(req.diff[i] + ch.recv_u64());
      }
      for (auto v : req.diff) ch.send_u64(v);
    }
    req.slot->ready.store(true);
    total_words += req.len;
  }
  requests_.clear();
  stats_.flushes += 1;
  stats_.opened_words += total_words;
  pending_words_ = 0;
  auto t1 = std::chrono::steady_clock::now();
  stats_.flush_ns += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

std::span<const int64_t> OpenCollector::view(const OpenHandle& h) const {
  if (!h.slot || !h.slot->ready.load()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  if (h.offset + h.len > h.slot->opened.size()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  return std::span<const int64_t>(h.slot->opened.data() + h.offset, h.len);
}

bool OpenCollector::ready(const OpenHandle& h) const {
  return h.slot && h.slot->ready.load() && h.offset + h.len <= h.slot->opened.size();
}

void OpenCollector::clear() {
  requests_.clear();
  pending_words_ = 0;
}

}  // namespace runtime

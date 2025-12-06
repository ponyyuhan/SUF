#include "runtime/open_collector.hpp"

#include <stdexcept>

namespace runtime {

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff) {
  size_t new_pending = pending_words_ + diff.size();
  if (limits_.max_pending_words > 0 && new_pending > limits_.max_pending_words) {
    throw std::runtime_error("OpenCollector: pending open budget exceeded");
  }
  OpenHandle h;
  h.offset = opened_.size();
  h.len = diff.size();
  Request r;
  r.diff = diff;
  r.offset = h.offset;
  r.len = h.len;
  requests_.push_back(std::move(r));
  opened_.resize(opened_.size() + diff.size(), 0);
  opened_valid_ = false;
  pending_words_ = new_pending;
  if (pending_words_ > stats_.max_pending_words) {
    stats_.max_pending_words = pending_words_;
  }
  return h;
}

void OpenCollector::flush(int party, net::Chan& ch) {
  if (requests_.empty()) return;
  size_t total_words = 0;
  for (const auto& req : requests_) {
    if (req.offset + req.len > opened_.size()) {
      throw std::runtime_error("OpenCollector: request out of range");
    }
    if (party == 0) {
      for (auto v : req.diff) ch.send_u64(v);
      for (size_t i = 0; i < req.len; ++i) {
        opened_[req.offset + i] = static_cast<int64_t>(req.diff[i] + ch.recv_u64());
      }
    } else {
      for (size_t i = 0; i < req.len; ++i) {
        opened_[req.offset + i] = static_cast<int64_t>(req.diff[i] + ch.recv_u64());
      }
      for (auto v : req.diff) ch.send_u64(v);
    }
    total_words += req.len;
  }
  requests_.clear();
  opened_valid_ = true;
  stats_.flushes += 1;
  stats_.opened_words += total_words;
  pending_words_ = 0;
}

std::span<const int64_t> OpenCollector::view(const OpenHandle& h) const {
  if (!opened_valid_ || h.offset + h.len > opened_.size()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  return std::span<const int64_t>(opened_.data() + h.offset, h.len);
}

bool OpenCollector::ready(const OpenHandle& h) const {
  return opened_valid_ && h.offset + h.len <= opened_.size();
}

void OpenCollector::clear() {
  requests_.clear();
  opened_.clear();
  opened_valid_ = false;
  pending_words_ = 0;
}

}  // namespace runtime

#include "runtime/open_collector.hpp"

#include <stdexcept>

namespace runtime {

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff) {
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
  return h;
}

void OpenCollector::flush(int party, net::Chan& ch) {
  if (requests_.empty()) return;
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
  }
  requests_.clear();
  opened_valid_ = true;
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
}

}  // namespace runtime

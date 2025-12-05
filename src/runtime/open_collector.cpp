#include "runtime/open_collector.hpp"

#include <stdexcept>

namespace runtime {

OpenHandle OpenCollector::enqueue(const std::vector<uint64_t>& diff) {
  OpenHandle h;
  h.offset = opened_.size();
  h.len = diff.size();
  requests_.push_back(Request{diff, h.offset, h.len});
  opened_.resize(opened_.size() + diff.size(), 0);
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
}

OpenView OpenCollector::view(const OpenHandle& h) const {
  if (h.offset + h.len > opened_.size()) {
    throw std::runtime_error("OpenCollector: view out of range");
  }
  OpenView v;
  v.data = opened_.data() + h.offset;
  v.len = h.len;
  return v;
}

void OpenCollector::clear() {
  requests_.clear();
  opened_.clear();
}

}  // namespace runtime

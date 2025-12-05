#include "runtime/pfss_superbatch.hpp"

#include <stdexcept>
#include <unordered_map>

namespace runtime {

PfssHandle PfssSuperBatch::enqueue_truncation(const compiler::TruncationLoweringResult& bundle,
                                              const gates::CompositePartyKey& key,
                                              gates::PostProcHook& hook,
                                              std::vector<uint64_t> hatx_public,
                                              nn::TensorView<uint64_t> out) {
  PreparedCompositeJob job;
  job.suf = &bundle.suf;
  job.key = &key;
  job.hook = &hook;
  job.hatx_public = std::move(hatx_public);
  job.out = out;
  job.token = completed_.size();
  completed_.push_back(CompletedJob{});
  jobs_.push_back(std::move(job));
  return PfssHandle{job.token};
}

PfssHandle PfssSuperBatch::enqueue_composite(PreparedCompositeJob job) {
  if (job.token == static_cast<size_t>(-1)) {
    job.token = completed_.size();
  }
  if (completed_.size() <= job.token) {
    completed_.resize(job.token + 1);
  }
  jobs_.push_back(std::move(job));
  return PfssHandle{jobs_.back().token};
}

void PfssSuperBatch::flush_and_finalize(int party,
                                        proto::PfssBackendBatch& backend,
                                        proto::IChannel& ch) {
  struct Key {
    const suf::SUF<uint64_t>* suf;
    const gates::CompositePartyKey* key;
    size_t r;
    size_t ell;
    bool operator==(const Key& o) const {
      return suf == o.suf && key == o.key && r == o.r && ell == o.ell;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<const void*>{}(k.suf) ^ (std::hash<const void*>{}(k.key) << 1) ^
             (std::hash<size_t>{}(k.r) << 2) ^ (std::hash<size_t>{}(k.ell) << 3);
    }
  };
  struct Slice {
    size_t job_idx;
    size_t start;
    size_t len;
  };
  std::unordered_map<Key, size_t, KeyHash> group_index;
  std::vector<std::vector<uint64_t>> grouped_hatx;
  std::vector<std::vector<Slice>> grouped_slices;

  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    auto& job = jobs_[idx];
    if (!job.suf || !job.key) {
      throw std::runtime_error("PfssSuperBatch: incomplete composite job");
    }
    size_t r = static_cast<size_t>(job.key->compiled.r);
    size_t ell = static_cast<size_t>(job.key->compiled.ell);
    Key k{job.suf, job.key, r, ell};
    size_t g;
    auto it = group_index.find(k);
    if (it == group_index.end()) {
      g = grouped_hatx.size();
      group_index.emplace(k, g);
      grouped_hatx.emplace_back();
      grouped_slices.emplace_back();
    } else {
      g = it->second;
    }
    size_t start = grouped_hatx[g].size();
    grouped_hatx[g].insert(grouped_hatx[g].end(), job.hatx_public.begin(), job.hatx_public.end());
    grouped_slices[g].push_back(Slice{idx, start, job.hatx_public.size()});
  }

  group_results_.clear();
  group_results_.reserve(group_index.size());

  for (auto& kv : group_index) {
    const auto& ktuple = kv.first;
    size_t g = kv.second;
    const auto* suf = ktuple.suf;
    const auto* key = ktuple.key;
    gates::CompositeBatchInput in{grouped_hatx[g].data(), grouped_hatx[g].size()};
    auto out = gates::composite_eval_batch_backend(party, backend, ch, *key, *suf, in);
    GroupResult& gr = group_results_.emplace_back();
    gr.suf = suf;
    gr.key = key;
    gr.r = ktuple.r;
    gr.ell = ktuple.ell;
    gr.arith = std::move(out.haty_share);
    gr.bools = std::move(out.bool_share);

    size_t r = gr.r;
    size_t ell = gr.ell;
    for (const auto& sl : grouped_slices[g]) {
      const auto& job = jobs_[sl.job_idx];
      if (job.token >= completed_.size()) continue;
      CompletedJob& cj = completed_[job.token];
      cj.r = r;
      cj.ell = ell;

      // Slice out this job's portion of the batch outputs.
      size_t arith_words = sl.len * r;
      size_t bool_words = sl.len * ell;
      const uint64_t* arith_base = gr.arith.data() + sl.start * r;
      const uint64_t* bool_base = ell > 0 ? (gr.bools.data() + sl.start * ell) : nullptr;

      std::vector<uint64_t> arith_slice(arith_base, arith_base + arith_words);
      std::vector<uint64_t> bool_slice;
      if (bool_base && bool_words > 0) {
        bool_slice.assign(bool_base, bool_base + bool_words);
      }

      if (job.hook) {
        proto::BeaverMul64 mul{party, ch, job.key->triples, 0};
        job.hook->configure(job.key->compiled.layout);
        std::vector<uint64_t> hooked(arith_words, 0);
        job.hook->run_batch(party, ch, mul,
                            job.hatx_public.data(),
                            arith_slice.data(), r,
                            bool_slice.data(), ell,
                            sl.len,
                            hooked.data());
        arith_slice.swap(hooked);
      }

      // Remove output masks and write to destination view if provided.
      cj.arith.resize(arith_slice.size());
      for (size_t i = 0; i < sl.len; ++i) {
        for (size_t rr = 0; rr < r; ++rr) {
          size_t idx = i * r + rr;
          uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
          uint64_t val = proto::sub_mod(arith_slice[idx], rout);
          cj.arith[idx] = val;
          if (idx < job.out.numel()) {
            job.out.data[idx] = val;
          }
        }
      }
      cj.bools = std::move(bool_slice);
    }
  }
  jobs_.clear();
}

PfssResultView PfssSuperBatch::view(const PfssHandle& h) const {
  if (h.token >= completed_.size()) return PfssResultView{};
  const auto& cj = completed_[h.token];
  PfssResultView v;
  v.arith = cj.arith.data();
  v.arith_words = cj.arith.size();
  v.bools = cj.bools.data();
  v.bool_words = cj.bools.size();
  v.r = cj.r;
  v.ell = cj.ell;
  return v;
}

void PfssSuperBatch::clear() {
  jobs_.clear();
  group_results_.clear();
  completed_.clear();
}

void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share) {
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;
  gates::PostProcHook* hook = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook) throw std::runtime_error("run_truncation_now: missing postproc hook");

  size_t n = x_share.size();
  std::vector<uint64_t> hatx_share(n);
  for (size_t i = 0; i < n; ++i) {
    hatx_share[i] = proto::add_mod(x_share[i], key.r_in_share);
  }

  // Open hatx (masked) to both parties.
  std::vector<uint64_t> other(n, 0);
  size_t byte_len = n * sizeof(uint64_t);
  if (party == 0) {
    ch.send_bytes(hatx_share.data(), byte_len);
    ch.recv_bytes(other.data(), byte_len);
  } else {
    ch.recv_bytes(other.data(), byte_len);
    ch.send_bytes(hatx_share.data(), byte_len);
  }
  std::vector<uint64_t> hatx_public(n);
  for (size_t i = 0; i < n; ++i) {
    hatx_public[i] = proto::add_mod(hatx_share[i], other[i]);
  }

  gates::CompositeBatchInput in{hatx_public.data(), n};
  auto out = gates::composite_eval_batch_with_postproc(
      party, backend, ch, key, bundle.suf, in, *hook);
  uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
  y_share.resize(n);
  for (size_t i = 0; i < n; ++i) {
    y_share[i] = proto::sub_mod(out.haty_share[i], r_out_share);
  }
}

}  // namespace runtime

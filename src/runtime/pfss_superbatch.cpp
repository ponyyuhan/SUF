#include "runtime/pfss_superbatch.hpp"

#include <stdexcept>
#include <unordered_map>
#include <random>
#include <algorithm>

namespace runtime {

PfssHandle PfssSuperBatch::enqueue_composite(PreparedCompositeJob job) {
  if (job.token == static_cast<size_t>(-1)) {
    job.token = completed_.size();
  }
  size_t hatx_words = job.hatx_public.size();
  size_t new_pending_jobs = pending_jobs_ + 1;
  size_t new_pending_hatx = pending_hatx_words_ + hatx_words;
  if (limits_.max_pending_jobs > 0 && new_pending_jobs > limits_.max_pending_jobs) {
    throw std::runtime_error("PfssSuperBatch: pending job limit exceeded");
  }
  if (limits_.max_pending_hatx_words > 0 && new_pending_hatx > limits_.max_pending_hatx_words) {
    throw std::runtime_error("PfssSuperBatch: pending hatx packing limit exceeded");
  }
  pending_jobs_ = new_pending_jobs;
  pending_hatx_words_ = new_pending_hatx;
  stats_.pending_jobs = pending_jobs_;
  stats_.pending_hatx = pending_hatx_words_;
  if (completed_.size() <= job.token) {
    completed_.resize(job.token + 1);
  }
  jobs_.push_back(std::move(job));
  flushed_ = false;
  return PfssHandle{jobs_.back().token};
}

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
  return enqueue_composite(std::move(job));
}

bool PfssSuperBatch::ready(const PfssHandle& h) const {
  return h.token < completed_.size() && !completed_[h.token].arith.empty();
}

void PfssSuperBatch::flush_eval(int party, proto::PfssBackendBatch& backend, proto::IChannel& ch) {
  if (limits_.max_flushes > 0 && stats_.flushes + 1 > limits_.max_flushes) {
    throw std::runtime_error("PfssSuperBatch: flush budget exceeded");
  }
  stats_.flushes += 1;
  stats_.jobs += jobs_.size();
  struct GroupKey {
    int r = 0;
    int ell = 0;
    int degree = 0;
    int pred_n = 0;
    int pred_out_mode = 0;
    int coeff_n = 0;
    int coeff_mode = 0;
    int coeff_words = 0;
    bool use_packed_pred = false;
    bool use_packed_cut = false;
    int packed_pred_words = 0;
    int packed_cut_words = 0;
    bool operator==(const GroupKey& o) const {
      return r == o.r && ell == o.ell && degree == o.degree &&
             pred_n == o.pred_n && pred_out_mode == o.pred_out_mode &&
             coeff_n == o.coeff_n && coeff_mode == o.coeff_mode && coeff_words == o.coeff_words &&
             use_packed_pred == o.use_packed_pred && use_packed_cut == o.use_packed_cut &&
             packed_pred_words == o.packed_pred_words && packed_cut_words == o.packed_cut_words;
    }
  };
  struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
      size_t h = std::hash<int>{}(k.r + (k.ell << 4) + (k.degree << 8));
      h ^= std::hash<int>{}(k.pred_n << 12) ^ std::hash<int>{}(k.pred_out_mode << 16);
      h ^= std::hash<int>{}(k.coeff_n << 20) ^ std::hash<int>{}(k.coeff_mode << 24);
      h ^= std::hash<int>{}(k.coeff_words << 28);
      h ^= std::hash<int>{}(k.packed_pred_words) ^ (std::hash<int>{}(k.packed_cut_words) << 1);
      h ^= std::hash<bool>{}(k.use_packed_pred) ^ (std::hash<bool>{}(k.use_packed_cut) << 2);
      return h;
    }
  };
  struct GroupData {
    std::vector<size_t> jobs;
  };
  struct BucketJob {
    size_t job_idx = 0;
    size_t start = 0;
    size_t len = 0;
  };
  struct Bucket {
    const gates::CompositePartyKey* key = nullptr;
    const suf::SUF<uint64_t>* suf = nullptr;
    std::vector<BucketJob> jobs;
    std::vector<uint64_t> hatx;
    size_t r = 0;
    size_t ell = 0;
  };

  std::unordered_map<GroupKey, size_t, GroupKeyHash> group_index;
  std::vector<GroupData> groups;

  group_results_.clear();
  slices_.assign(jobs_.size(), JobSlice{});
  size_t total_arith_words = 0;
  size_t total_bool_words = 0;

  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    auto& job = jobs_[idx];
    if (!job.suf || !job.key) {
      throw std::runtime_error("PfssSuperBatch: incomplete composite job");
    }
    const auto& comp = job.key->compiled;
    GroupKey gk;
    gk.r = comp.r;
    gk.ell = comp.ell;
    gk.degree = comp.degree;
    gk.pred_n = comp.pred.n;
    gk.pred_out_mode = static_cast<int>(comp.pred.out_mode);
    gk.coeff_n = comp.coeff.n;
    gk.coeff_mode = static_cast<int>(comp.coeff.mode);
    gk.coeff_words = comp.coeff.out_words;
    gk.use_packed_pred = job.key->use_packed_pred;
    gk.use_packed_cut = job.key->use_packed_cut;
    gk.packed_pred_words = job.key->packed_pred_words;
    gk.packed_cut_words = job.key->packed_cut_words;
    size_t g_idx;
    auto it = group_index.find(gk);
    if (it == group_index.end()) {
      g_idx = groups.size();
      group_index.emplace(gk, g_idx);
      groups.emplace_back();
    } else {
      g_idx = it->second;
    }
    groups[g_idx].jobs.push_back(idx);
  }

  for (const auto& gd : groups) {
    std::unordered_map<const gates::CompositePartyKey*, size_t> bucket_index;
    std::vector<Bucket> buckets;
    for (auto job_idx : gd.jobs) {
      auto& job = jobs_[job_idx];
      size_t bidx;
      auto it = bucket_index.find(job.key);
      if (it == bucket_index.end()) {
        bidx = buckets.size();
        bucket_index.emplace(job.key, bidx);
        Bucket b;
        b.key = job.key;
        b.suf = job.suf;
        b.r = static_cast<size_t>(job.key->compiled.r);
        b.ell = static_cast<size_t>(job.key->compiled.ell);
        buckets.push_back(std::move(b));
      } else {
        bidx = it->second;
      }
      Bucket& b = buckets[bidx];
      BucketJob bj;
      bj.job_idx = job_idx;
      bj.start = b.hatx.size();
      bj.len = job.hatx_public.size();
      b.hatx.insert(b.hatx.end(), job.hatx_public.begin(), job.hatx_public.end());
      b.jobs.push_back(bj);
    }

    for (auto& b : buckets) {
      stats_.max_bucket_hatx = std::max(stats_.max_bucket_hatx, b.hatx.size());
      stats_.max_bucket_jobs = std::max(stats_.max_bucket_jobs, b.jobs.size());
      gates::CompositeBatchInput in{b.hatx.data(), b.hatx.size()};
      auto out = gates::composite_eval_batch_backend(party, backend, ch, *b.key, *b.suf, in);
      size_t gr_idx = group_results_.size();
      GroupResult gr;
      gr.suf = b.suf;
      gr.key = b.key;
      gr.r = b.r;
      gr.ell = b.ell;
      gr.arith = std::move(out.haty_share);
      gr.bools = std::move(out.bool_share);
      total_arith_words += gr.arith.size();
      total_bool_words += gr.bools.size();
      for (const auto& bj : b.jobs) {
        if (bj.job_idx >= slices_.size()) continue;
        slices_[bj.job_idx].group_result = gr_idx;
        slices_[bj.job_idx].start = bj.start;
        slices_[bj.job_idx].len = bj.len;
      }
      group_results_.push_back(std::move(gr));
    }
  }
  flushed_ = true;
  stats_.arith_words += total_arith_words;
  stats_.pred_bits += total_bool_words * 64;
  pending_jobs_ = 0;
  pending_hatx_words_ = 0;
  stats_.pending_jobs = 0;
  stats_.pending_hatx = 0;
  populate_completed_();
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

void PfssSuperBatch::populate_completed_() {
  // Populate per-job completed slices (without running hooks).
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    if (idx >= completed_.size()) continue;
    const auto& job = jobs_[idx];
    const auto& gr = group_results_[sl.group_result];
    size_t r = gr.r;
    size_t ell = gr.ell;
    size_t arith_words = sl.len * r;
    size_t bool_words = sl.len * ell;
    const uint64_t* arith_base = gr.arith.data() + sl.start * r;
    const uint64_t* bool_base =
        (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

    CompletedJob& cj = completed_[job.token];
    cj.r = r;
    cj.ell = ell;
    cj.arith.assign(arith_base, arith_base + arith_words);
    if (bool_base && bool_words > 0) {
      cj.bools.assign(bool_base, bool_base + bool_words);
    } else {
      cj.bools.clear();
    }
  }
}

void PfssSuperBatch::clear() {
  jobs_.clear();
  group_results_.clear();
  completed_.clear();
  slices_.clear();
  flushed_ = false;
  pending_jobs_ = 0;
  pending_hatx_words_ = 0;
  stats_.pending_jobs = 0;
  stats_.pending_hatx = 0;
}

void PfssSuperBatch::finalize_all(int party, proto::IChannel& ch) {
  if (!flushed_) return;
  if (completed_.empty()) populate_completed_();
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    const auto& job = jobs_[idx];
    if (job.token >= completed_.size()) continue;
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    const auto& gr = group_results_[sl.group_result];

    size_t r = gr.r;
    size_t ell = gr.ell;
    size_t arith_words = sl.len * r;
    size_t bool_words = sl.len * ell;
    const uint64_t* arith_base = gr.arith.data() + sl.start * r;
    const uint64_t* bool_base =
        (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

    std::vector<uint64_t> arith_slice(arith_base, arith_base + arith_words);
    std::vector<uint64_t> bool_slice;
    if (bool_base && bool_words > 0) {
      bool_slice.assign(bool_base, bool_base + bool_words);
    }

    if (job.hook) {
      // Always use a deterministic synthetic triple pool sized generously; avoids triple exhaustion
      // in hooks regardless of key provisioning.
      size_t generous_need = std::max<size_t>(512, job.hatx_public.size() * 256);
      std::vector<proto::BeaverTriple64Share> synth_triples;
      synth_triples.reserve(generous_need);
      std::mt19937_64 rng(job.key->compiled.r_in ^ 0x70667373u);
      for (size_t i = 0; i < generous_need; ++i) {
        uint64_t a = rng();
        uint64_t b = rng();
        uint64_t c = proto::mul_mod(a, b);
        uint64_t a0 = rng();
        uint64_t a1 = a - a0;
        uint64_t b0 = rng();
        uint64_t b1 = b - b0;
        uint64_t c0 = rng();
        uint64_t c1 = c - c0;
        synth_triples.push_back((party == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                                             : proto::BeaverTriple64Share{a1, b1, c1});
      }
      proto::BeaverMul64 mul{party, ch, synth_triples, 0};
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

    CompletedJob& cj = completed_[job.token];
    cj.r = r;
    cj.ell = ell;
    cj.arith.resize(arith_slice.size());
      for (size_t i = 0; i < sl.len; ++i) {
        for (size_t rr = 0; rr < r; ++rr) {
          size_t out_idx = i * r + rr;
          uint64_t val = arith_slice[out_idx];
          uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
          val = proto::sub_mod(val, rout);
          cj.arith[out_idx] = val;
          if (out_idx < job.out.numel() && job.out.data) {
            job.out.data[out_idx] = val;
          }
        }
    }
    cj.bools = std::move(bool_slice);
  }
  jobs_.clear();
  slices_.clear();
  group_results_.clear();
  flushed_ = false;
}

void PfssSuperBatch::flush_and_finalize(int party,
                                        proto::PfssBackendBatch& backend,
                                        proto::IChannel& ch) {
  flush_eval(party, backend, ch);
  populate_completed_();
  finalize_all(party, ch);
  clear();
}

void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share) {
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;
  const gates::PostProcHook* hook_ptr = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook_ptr) {
    throw std::runtime_error("run_truncation_now: missing truncation hook");
  }
  gates::PostProcHook* hook = const_cast<gates::PostProcHook*>(hook_ptr);
  hook->configure(key.compiled.layout);
  size_t N = x_share.size();
  std::vector<uint64_t> hatx_share(N);
  for (size_t i = 0; i < N; ++i) {
    uint64_t rin = (!key.r_in_share_vec.empty() && key.r_in_share_vec.size() > i)
                       ? key.r_in_share_vec[i]
                       : key.r_in_share;
    hatx_share[i] = proto::add_mod(x_share[i], rin);
  }
  std::vector<uint64_t> other(N, 0);
  if (party == 0) {
    ch.send_bytes(hatx_share.data(), N * sizeof(uint64_t));
    ch.recv_bytes(other.data(), N * sizeof(uint64_t));
  } else {
    ch.recv_bytes(other.data(), N * sizeof(uint64_t));
    ch.send_bytes(hatx_share.data(), N * sizeof(uint64_t));
  }
  std::vector<uint64_t> hatx_public(N);
  for (size_t i = 0; i < N; ++i) {
    hatx_public[i] = proto::add_mod(hatx_share[i], other[i]);
  }
  gates::CompositeBatchInput in{hatx_public.data(), N};
  auto out = gates::composite_eval_batch_with_postproc(
      party, backend, ch, key, bundle.suf, in, *hook);
  uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
  y_share.resize(N);
  for (size_t i = 0; i < N; ++i) {
    y_share[i] = proto::sub_mod(out.haty_share[i], r_out_share);
  }
}

}  // namespace runtime

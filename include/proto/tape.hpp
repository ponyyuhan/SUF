#pragma once

#include "proto/common.hpp"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace proto {

using u32 = uint32_t;

// Minimal tag set required by milestone tapes.
enum class TapeTag : u32 {
  kU64      = 1,
  kU64Vec   = 2,
  kBytes    = 3,
  kTriple64 = 4
};

struct Triple64Share { u64 a = 0, b = 0, c = 0; };

// ------------------------------ sinks/sources ------------------------------
struct ITapeSink {
  virtual ~ITapeSink() = default;
  virtual void write(std::span<const u8> bytes) = 0;
  virtual void flush() {}
};

struct ITapeSource {
  virtual ~ITapeSource() = default;
  virtual void read_exact(std::span<u8> out) = 0;
  virtual bool eof() const = 0;
};

class VecTapeSink final : public ITapeSink {
public:
  std::vector<u8> buf;
  void write(std::span<const u8> bytes) override {
    buf.insert(buf.end(), bytes.begin(), bytes.end());
  }
};

class VecTapeSource final : public ITapeSource {
public:
  explicit VecTapeSource(const std::vector<u8>& b) : buf_(b) {}

  void read_exact(std::span<u8> out) override {
    if (off_ + out.size() > buf_.size()) throw std::runtime_error("tape: read past end");
    std::memcpy(out.data(), buf_.data() + off_, out.size());
    off_ += out.size();
  }

  bool eof() const override { return off_ >= buf_.size(); }
  std::size_t offset() const { return off_; }

private:
  const std::vector<u8>& buf_;
  std::size_t off_ = 0;
};

class FileTapeSink final : public ITapeSink {
public:
  explicit FileTapeSink(const std::string& path)
      : out_(path, std::ios::binary | std::ios::trunc) {
    if (!out_) throw std::runtime_error("tape: open output failed");
  }

  void write(std::span<const u8> bytes) override {
    out_.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!out_) throw std::runtime_error("tape: file write failed");
  }

  void flush() override {
    out_.flush();
    if (!out_) throw std::runtime_error("tape: file flush failed");
  }

private:
  std::ofstream out_;
};

class FileTapeSource final : public ITapeSource {
public:
  explicit FileTapeSource(const std::string& path)
      : in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("tape: open input failed");
  }

  void read_exact(std::span<u8> out) override {
    in_.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(out.size()));
    if (!in_) throw std::runtime_error("tape: file read failed");
  }

  bool eof() const override {
    return in_.peek() == std::char_traits<char>::eof();
  }

private:
  mutable std::ifstream in_;
};

// ------------------------------ Writer ------------------------------
class TapeWriter {
public:
  TapeWriter() : owned_(std::make_unique<VecTapeSink>()), sink_(owned_.get()) {}
  explicit TapeWriter(ITapeSink& sink) : sink_(&sink) {}

  void write_u64(u64 v) { write_record(TapeTag::kU64, std::span<const u8>(reinterpret_cast<const u8*>(&v), 8)); }

  void write_bytes(std::span<const u8> b) { write_record(TapeTag::kBytes, b); }
  void write_bytes(const std::vector<u8>& b) { write_bytes(std::span<const u8>(b.data(), b.size())); }

  void write_u64_vec(std::span<const u64> v) {
    write_record(TapeTag::kU64Vec, std::span<const u8>(reinterpret_cast<const u8*>(v.data()), v.size() * sizeof(u64)));
  }
  void write_u64_vec(const std::vector<u64>& v) { write_u64_vec(std::span<const u64>(v.data(), v.size())); }

  template<typename Triple>
  void write_triple64(const Triple& t) {
    write_record(TapeTag::kTriple64, std::span<const u8>(reinterpret_cast<const u8*>(&t), sizeof(Triple)));
  }

  template<typename Triple>
  void write_triple64_vec(std::span<const Triple> v) {
    write_record(TapeTag::kTriple64, std::span<const u8>(reinterpret_cast<const u8*>(v.data()), v.size() * sizeof(Triple)));
  }

  void flush() { sink_->flush(); }

  const std::vector<u8>& data() const {
    if (!owned_) throw std::runtime_error("tape: data() only valid for in-memory writer");
    return owned_->buf;
  }

private:
  std::unique_ptr<VecTapeSink> owned_;
  ITapeSink* sink_ = nullptr;

  void write_record(TapeTag tag, std::span<const u8> payload) {
    u32 t = static_cast<u32>(tag);
    u32 len = static_cast<u32>(payload.size());
    u8 header[8];
    std::memcpy(header, &t, 4);
    std::memcpy(header + 4, &len, 4);
    sink_->write(std::span<const u8>(header, 8));
    if (!payload.empty()) sink_->write(payload);
  }
};

// ------------------------------ Reader ------------------------------
class TapeReader {
public:
  explicit TapeReader(ITapeSource& src) : src_(&src) {}
  explicit TapeReader(const std::vector<u8>& buf)
      : owned_source_(std::make_unique<VecTapeSource>(buf)) {
    src_ = owned_source_.get();
  }

  u64 read_u64() {
    auto payload = read_record(TapeTag::kU64);
    if (payload.size() != 8) throw std::runtime_error("tape: read_u64 size mismatch");
    u64 v;
    std::memcpy(&v, payload.data(), 8);
    return v;
  }

  std::vector<u8> read_bytes() { return read_record(TapeTag::kBytes); }

  std::vector<u64> read_u64_vec() {
    auto payload = read_record(TapeTag::kU64Vec);
    if (payload.size() % 8 != 0) throw std::runtime_error("tape: read_u64_vec size mismatch");
    std::vector<u64> v(payload.size() / 8);
    std::memcpy(v.data(), payload.data(), payload.size());
    return v;
  }

  std::vector<u64> read_u64_vec(size_t expect_words) {
    auto v = read_u64_vec();
    if (expect_words != 0 && v.size() != expect_words) throw std::runtime_error("tape: read_u64_vec unexpected length");
    return v;
  }

  template<typename Triple>
  Triple read_triple64() {
    auto payload = read_record(TapeTag::kTriple64);
    if (payload.size() != sizeof(Triple)) throw std::runtime_error("tape: read_triple64 size mismatch");
    Triple t;
    std::memcpy(&t, payload.data(), sizeof(Triple));
    return t;
  }

  template<typename Triple>
  std::vector<Triple> read_triple64_vec() {
    auto payload = read_record(TapeTag::kTriple64);
    if (payload.size() % sizeof(Triple) != 0) throw std::runtime_error("tape: read_triple64_vec size mismatch");
    size_t n = payload.size() / sizeof(Triple);
    std::vector<Triple> v(n);
    std::memcpy(v.data(), payload.data(), payload.size());
    return v;
  }

  bool eof() const { return src_->eof(); }

private:
  ITapeSource* src_;
  std::unique_ptr<VecTapeSource> owned_source_;

  std::vector<u8> read_record(TapeTag expect) {
    u8 header[8];
    src_->read_exact(std::span<u8>(header, 8));
    u32 tag = 0, len = 0;
    std::memcpy(&tag, header, 4);
    std::memcpy(&len, header + 4, 4);
    if (tag != static_cast<u32>(expect)) throw std::runtime_error("tape: tag mismatch");
    std::vector<u8> payload(len);
    if (len) src_->read_exact(std::span<u8>(payload.data(), payload.size()));
    return payload;
  }
};

}  // namespace proto

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <openssl/evp.h>

// Minimal SHA256 adapter for libdpf.
//
// libdpf's public headers include `hash-library/sha256.cpp` as a header. The
// upstream repository's hash-library submodule is not always present when
// fetched via CMake/FetchContent, so we provide a tiny OpenSSL-backed shim with
// the expected `SHA256` interface.
class SHA256 {
 public:
  SHA256() {
    ctx_ = EVP_MD_CTX_new();
    if (!ctx_) throw std::runtime_error("SHA256: EVP_MD_CTX_new failed");
    if (EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) != 1) {
      EVP_MD_CTX_free(ctx_);
      ctx_ = nullptr;
      throw std::runtime_error("SHA256: EVP_DigestInit_ex failed");
    }
  }

  ~SHA256() {
    if (ctx_) EVP_MD_CTX_free(ctx_);
  }

  SHA256(const SHA256&) = delete;
  SHA256& operator=(const SHA256&) = delete;

  void add(const void* data, std::size_t len) {
    if (!ctx_) throw std::runtime_error("SHA256: null ctx");
    if (len == 0) return;
    if (EVP_DigestUpdate(ctx_, data, len) != 1) {
      throw std::runtime_error("SHA256: EVP_DigestUpdate failed");
    }
  }

  void getHash(unsigned char* out) {
    if (!ctx_) throw std::runtime_error("SHA256: null ctx");
    if (!out) throw std::runtime_error("SHA256: null output");
    EVP_MD_CTX* tmp = EVP_MD_CTX_new();
    if (!tmp) throw std::runtime_error("SHA256: EVP_MD_CTX_new (tmp) failed");
    if (EVP_MD_CTX_copy_ex(tmp, ctx_) != 1) {
      EVP_MD_CTX_free(tmp);
      throw std::runtime_error("SHA256: EVP_MD_CTX_copy_ex failed");
    }
    unsigned int out_len = 0;
    if (EVP_DigestFinal_ex(tmp, out, &out_len) != 1) {
      EVP_MD_CTX_free(tmp);
      throw std::runtime_error("SHA256: EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(tmp);
    if (out_len != 32) {
      throw std::runtime_error("SHA256: unexpected digest length");
    }
  }

 private:
  EVP_MD_CTX* ctx_ = nullptr;
};


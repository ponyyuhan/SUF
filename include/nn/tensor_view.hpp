#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace nn {

template<typename T, size_t MaxDims = 4>
struct TensorView {
  T* data = nullptr;
  std::array<size_t, MaxDims> shape{};
  std::array<size_t, MaxDims> stride{};
  size_t dims = 0;

  TensorView() = default;

  TensorView(T* d, std::initializer_list<size_t> shp) : data(d), dims(shp.size()) {
    size_t i = 0;
    size_t prod = 1;
    for (auto it = shp.end(); it != shp.begin();) {
      --it;
      shape[dims - 1 - i] = *it;
      stride[dims - 1 - i] = prod;
      prod *= *it;
      ++i;
    }
  }

  size_t numel() const {
    size_t n = 1;
    for (size_t i = 0; i < dims; ++i) n *= shape[i];
    return n;
  }

  size_t offset(const std::array<size_t, MaxDims>& idx) const {
    size_t off = 0;
    for (size_t i = 0; i < dims; ++i) off += idx[i] * stride[i];
    return off;
  }

  T& operator[](size_t i) const { return data[i]; }
};

template<typename T>
inline TensorView<T> view3(T* data, size_t a, size_t b, size_t c) {
  TensorView<T> v;
  v.data = data;
  v.dims = 3;
  v.shape = {a, b, c, 1};
  v.stride = {b * c, c, 1, 1};
  return v;
}

template<typename T>
inline TensorView<T> view2(T* data, size_t a, size_t b) {
  TensorView<T> v;
  v.data = data;
  v.dims = 2;
  v.shape = {a, b, 1, 1};
  v.stride = {b, 1, 1, 1};
  return v;
}

template<typename T>
inline TensorView<T> view4(T* data, size_t a, size_t b, size_t c, size_t d) {
  TensorView<T> v;
  v.data = data;
  v.dims = 4;
  v.shape = {a, b, c, d};
  v.stride = {b * c * d, c * d, d, 1};
  return v;
}

template<typename T, size_t MaxDims>
inline TensorView<T, MaxDims> reshape(TensorView<T, MaxDims> base,
                                      std::initializer_list<size_t> shp) {
  TensorView<T, MaxDims> v;
  v.data = base.data;
  v.dims = shp.size();
  size_t idx = 0;
  size_t prod = 1;
  for (auto it = shp.end(); it != shp.begin();) {
    --it;
    v.shape[v.dims - 1 - idx] = *it;
    v.stride[v.dims - 1 - idx] = prod;
    prod *= *it;
    ++idx;
  }
  return v;
}

template<typename T, size_t MaxDims>
inline TensorView<T, MaxDims> slice(TensorView<T, MaxDims> base,
                                    size_t dim,
                                    size_t start,
                                    size_t len) {
  TensorView<T, MaxDims> v = base;
  size_t offset = start * base.stride[dim];
  v.data = base.data + offset;
  v.shape[dim] = len;
  return v;
}

template<typename T, size_t MaxDims>
inline TensorView<T, MaxDims> transpose_view(TensorView<T, MaxDims> base,
                                             size_t dim0,
                                             size_t dim1) {
  TensorView<T, MaxDims> v = base;
  std::swap(v.shape[dim0], v.shape[dim1]);
  std::swap(v.stride[dim0], v.stride[dim1]);
  return v;
}

}  // namespace nn

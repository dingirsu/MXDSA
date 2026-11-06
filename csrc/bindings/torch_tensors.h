#pragma once

#include <functional>

#include <torch/python.h>

#include "common/host.h"

template <typename T>
static inline bool check_optional_tensor(const T& tensor_or_opt,
                                         const std::function<bool(const at::Tensor&)>& check_fn) {
    if constexpr (std::is_same<T, at::Tensor>::value) {
        return check_fn(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return check_fn(tensor_or_opt.value());
        } else {
            return true;
        }
    }
}

template <typename PtrT> static inline PtrT* get_tensor_ptr(const at::Tensor& tensor) {
    if (tensor.has_storage()) {
        return (PtrT*)tensor.data_ptr();
    } else {
        return nullptr;
    }
}

template <typename PtrT, typename T>
static inline PtrT* get_optional_tensor_ptr(const T& tensor_or_opt) {
    if constexpr (std::is_same<T, at::Tensor>::value) {
        return get_tensor_ptr<PtrT>(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return get_tensor_ptr<PtrT>(*tensor_or_opt);
        } else {
            return nullptr;
        }
    }
}

#define CHECK_DEVICE(tensor)                                                                       \
    TORCH_CHECK(check_optional_tensor(tensor, [](const at::Tensor& t) { return t.is_cuda(); }),    \
                #tensor " must be on CUDA")

#define CHECK_NDIM(tensor, ndim)                                                                   \
    TORCH_CHECK(                                                                                   \
        check_optional_tensor(tensor, [&](const at::Tensor& t) { return t.dim() == (ndim); }),     \
        #tensor " must have " #ndim " dimensions")

#define CHECK_SHAPE(tensor, ...)                                                                   \
    TORCH_CHECK(check_optional_tensor(tensor,                                                      \
                                      [&](const at::Tensor& t) {                                   \
                                          return t.sizes() == torch::IntArrayRef({__VA_ARGS__});   \
                                      }),                                                          \
                #tensor " must have shape (" #__VA_ARGS__ ")")

#define CHECK_CONTIGUOUS(tensor)                                                                   \
    TORCH_CHECK(                                                                                   \
        check_optional_tensor(tensor, [](const at::Tensor& t) { return t.is_contiguous(); }),      \
        #tensor " must be contiguous")

#define CHECK_LAST_DIM_CONTIGUOUS(tensor)                                                          \
    TORCH_CHECK(                                                                                   \
        check_optional_tensor(                                                                     \
            tensor, [](const at::Tensor& t) { return t.size(-1) == 1 || t.stride(-1) == 1; }),     \
        #tensor " must have contiguous last dimension")

#define CHECK_DTYPE(tensor, target_dtype)                                                          \
    TORCH_CHECK(check_optional_tensor(                                                             \
                    tensor, [](const at::Tensor& t) { return t.dtype() == (target_dtype); }),      \
                #tensor " must have dtype " #target_dtype)

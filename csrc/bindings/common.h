#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

static constexpr float LOG_2_E = 1.44269504f;

inline int int64_stride_to_int(int64_t stride) {
    TORCH_CHECK(stride <= std::numeric_limits<int>::max(), "Stride exceeds int32 limit: ", stride);
    return static_cast<int>(stride);
}

template <typename Check>
inline bool check_optional_tensor(const at::Tensor& tensor, const Check& check) {
    return check(tensor);
}

template <typename Check>
inline bool check_optional_tensor(const std::optional<at::Tensor>& tensor, const Check& check) {
    return !tensor || check(*tensor);
}

template <typename T> inline T* get_optional_tensor_ptr(const std::optional<at::Tensor>& tensor) {
    return tensor && tensor->has_storage() ? static_cast<T*>(tensor->data_ptr()) : nullptr;
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

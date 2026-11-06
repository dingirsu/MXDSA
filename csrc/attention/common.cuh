#pragma once

#include <math_constants.h>

#include "ptx/math.cuh"
#include "ptx/memory.cuh"
#include "ptx/mma.cuh"
#include "ptx/tmem.cuh"
namespace sm100 {
template <int D_V, int CHUNK_SIZE, int TMEM_COL_START> DEVICE void rescale_O(float scale_factor) {
    float2 scale_factor_float2 = {scale_factor, scale_factor};
    float2 o[CHUNK_SIZE / 2];

    UNROLL
    for (int chunk_idx = 0; chunk_idx < (D_V / 2) / CHUNK_SIZE; ++chunk_idx) {
        // Load O
        ptx::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx * CHUNK_SIZE, o);
        ptx::fence_view_async_tmem_load();

        // Mult
        for (int i = 0; i < CHUNK_SIZE / 2; ++i) {
            o[i] = ptx::float2_mul(o[i], scale_factor_float2);
        }

        // Store O
        ptx::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx * CHUNK_SIZE, o);
        ptx::fence_view_async_tmem_store();
    }
}

template <int NUM_ELEMS_PER_THREAD> DEVICE float get_max(float p[NUM_ELEMS_PER_THREAD]) {
    float local_max = -CUDART_INF_F;
    UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
        local_max = max(local_max, p[i]);
    }
    return local_max;
}


DEVICE
char load_indices_and_generate_mask(int lane_idx, int* gIndices, int s_kv, int abs_pos_start,
                                    int topk_length) {
    int indices[8];
    KU_LDG_256(gIndices + lane_idx * 8, indices, ".nc", "no_allocate", "evict_normal", "256B");
    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
        int abs_pos = abs_pos_start + lane_idx * 8 + rel_pos_in_lane;
        return index >= 0 && index < s_kv && abs_pos < topk_length;
    };
    char is_ks_valid_mask = is_valid(7, indices[7]) << 7 | is_valid(6, indices[6]) << 6 |
                            is_valid(5, indices[5]) << 5 | is_valid(4, indices[4]) << 4 |
                            is_valid(3, indices[3]) << 3 | is_valid(2, indices[2]) << 2 |
                            is_valid(1, indices[1]) << 1 | is_valid(0, indices[0]) << 0;
    return is_ks_valid_mask;
}

DEVICE void copy_k_scale_smem_to_tmem(void* ptr, uint32_t col) {
    uint64_t desc = 0x0000400800000000ull | (ptx::cast_smem_ptr_to_uint(ptr) >> 4);
    ptx::SM100_UTCCP_2x64dp128bitlw0123_2cta::copy(desc, col);
}

DEVICE void store_s_scale_to_tmem(uint32_t tmem_col, int warp_in_warpgroup, uint8_t scale0_bits,
                                  uint8_t scale1_bits) {
    // S is quantized independently for each softmax row.  The O MMA consumes
    // four K=32 SFA bytes from one 32-bit TMEM word, so repeat this row's
    // UE8M0 factor across the word and keep the duplicated 2x2 DP layout.
    const uint32_t packed_scale = uint32_t(scale0_bits) | (uint32_t(scale1_bits) << 8) |
                                  (uint32_t(scale0_bits) << 16) | (uint32_t(scale1_bits) << 24);
    const uint32_t warp_dp_addr = tmem_col + warp_in_warpgroup * 32 * 65536;
    UNROLL
    for (int row_group = 0; row_group < 4; ++row_group) {
        ptx::SM100_TMEM_STORE_32dp32b1x::copy(packed_scale, warp_dp_addr + row_group);
    }
}

// Expansion of UMMA Layout_K_SW{128,64}/INTER atoms, in elements.
// Tiling repeats the K atom after all MN rows. Swizzle<B,4,3> acts in bytes.
template <int MN, int SW, class T> DEVICE T* k_major_ptr(T* base, int row, int col) {
    constexpr int atom = SW ? SW : 16;
    int offset = (col * sizeof(T)) % atom + row * atom + (col * sizeof(T) / atom) * MN * atom;
    uintptr_t b = reinterpret_cast<uintptr_t>(base) + offset;
    if constexpr (SW > 16)
        b ^= (b >> 3) & (SW - 16);
    return reinterpret_cast<T*>(b);
}
// Descriptor constants and fragment strides are checked by tests/layout_probe.cu.
// The low 14 bits contain the shared byte address / 16; offsets below are in that unit.
template <int N, bool Decode>
DEVICE void issue_qk(const void* q, const void* k, uint32_t c, uint32_t sa, uint32_t sb) {
    uint64_t a = 0x4000404000010000ull | ((ptx::cast_smem_ptr_to_uint(q) >> 4) & 0x3fff);
    uint64_t b = 0x4000404000010000ull | ((ptx::cast_smem_ptr_to_uint(k) >> 4) & 0x3fff);
    UNROLL
    for (int kk = 0; kk < 16; kk++) {
        uint32_t idesc =
            (8u << 24) | (1u << 23) | ((N / 8) << 17) | ((kk & 3u) << 29) | (((kk / 2) & 3u) << 4);
        ptx::mma_mxfp8(a + (kk % 2) * 4 + ((kk / 2) % 4) * 512 + (kk / 8) * 2,
                  b + (kk % 4) * 2 + (kk / 4) * (N * 4), c + (Decode && kk >= 8 ? N / 2 : 0), idesc,
                  sa + 2 * (kk / 4), sb + 2 * (kk / 8), !(kk == 0 || (Decode && kk == 8)));
    }
}
template <int N, bool Decode>
DEVICE void issue_sv(const void* s, const void* v, uint32_t c, uint32_t sa, uint32_t sb,
                     bool clear) {
    uint64_t a = (Decode ? 0x0000400800400000ull : 0x8000402000010000ull) |
                 ((ptx::cast_smem_ptr_to_uint(s) >> 4) & 0x3fff);
    uint64_t b = 0x4000404000000000ull | ((ptx::cast_smem_ptr_to_uint(v) >> 4) & 0x3fff);
    UNROLL
    for (int kk = 0; kk < N / 32; kk++) {
        UNROLL
        for (int n = 0; n < 2; n++) {
            uint32_t idesc = (8u << 24) | (1u << 23) | (32u << 17) | (1u << 16) |
                             ((kk & 3u) << 29) | ((kk & 3u) << 4);
            ptx::mma_mxfp8(a + (Decode ? kk * 128 : (kk % 2) * 2 + (kk / 2) * 256),
                      b + kk * 256 + n * N * 8, c + n * 128, idesc, sa, sb + n * 4,
                      !(clear && kk == 0));
        }
    }
}
} // namespace sm100

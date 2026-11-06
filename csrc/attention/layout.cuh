#pragma once
#include "ptx/mma.cuh"
namespace ptx {
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
    uint64_t a = 0x4000404000010000ull | ((cast_smem_ptr_to_uint(q) >> 4) & 0x3fff);
    uint64_t b = 0x4000404000010000ull | ((cast_smem_ptr_to_uint(k) >> 4) & 0x3fff);
    UNROLL
    for (int kk = 0; kk < 16; kk++) {
        uint32_t idesc =
            (8u << 24) | (1u << 23) | ((N / 8) << 17) | ((kk & 3u) << 29) | (((kk / 2) & 3u) << 4);
        mma_mxfp8(a + (kk % 2) * 4 + ((kk / 2) % 4) * 512 + (kk / 8) * 2,
                  b + (kk % 4) * 2 + (kk / 4) * (N * 4), c + (Decode && kk >= 8 ? N / 2 : 0), idesc,
                  sa + 2 * (kk / 4), sb + 2 * (kk / 8), !(kk == 0 || (Decode && kk == 8)));
    }
}
template <int N, bool Decode>
DEVICE void issue_sv(const void* s, const void* v, uint32_t c, uint32_t sa, uint32_t sb,
                     bool clear) {
    uint64_t a = (Decode ? 0x0000400800400000ull : 0x8000402000010000ull) |
                 ((cast_smem_ptr_to_uint(s) >> 4) & 0x3fff);
    uint64_t b = 0x4000404000000000ull | ((cast_smem_ptr_to_uint(v) >> 4) & 0x3fff);
    UNROLL
    for (int kk = 0; kk < N / 32; kk++) {
        UNROLL
        for (int n = 0; n < 2; n++) {
            uint32_t idesc = (8u << 24) | (1u << 23) | (32u << 17) | (1u << 16) |
                             ((kk & 3u) << 29) | ((kk & 3u) << 4);
            mma_mxfp8(a + (Decode ? kk * 128 : (kk % 2) * 2 + (kk / 2) * 256),
                      b + kk * 256 + n * N * 8, c + n * 128, idesc, sa, sb + n * 4,
                      !(clear && kk == 0));
        }
    }
}
} // namespace ptx

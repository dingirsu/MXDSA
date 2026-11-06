#pragma once

// Tensor-memory allocation, shared-to-TMEM copies, loads/stores and waits.
#include "ptx/common.cuh"

namespace ptx {

struct Allocator2Sm {
    DEVICE void allocate(int n, uint32_t* p) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                         cast_smem_ptr_to_uint(p)),
                     "r"(n));
    }
    DEVICE void free(uint32_t p, int n) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(p), "r"(n));
    }
    DEVICE void release_allocation_lock() {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
};
struct SM100_UTCCP_2x64dp128bitlw0123_2cta {
    using SRegisters = uint64_t[1];
    using DRegisters = uint32_t[1];

    DEVICE static void copy(uint64_t const& src_addr, uint32_t const& dst_addr) {
        asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
                     :
                     : "r"(dst_addr) "l"(src_addr));
    }
};
template <int N> DEVICE void tmem_ld_32dp32bNx(uint32_t addr, void* dst);
template <int N> DEVICE void tmem_st_32dp32bNx(uint32_t addr, const void* src);
template <> DEVICE void tmem_ld_32dp32bNx<1>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(p[0]) : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<1>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%1], {%0};" ::"r"(p[0]), "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<2>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
                 : "=r"(p[0]), "=r"(p[1])
                 : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<2>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0, %1};" ::"r"(p[0]), "r"(p[1]),
                 "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<4>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3])
                 : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<4>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0, %1, %2, %3};" ::"r"(p[0]),
                 "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<8>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]), "=r"(p[5]),
                   "=r"(p[6]), "=r"(p[7])
                 : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<8>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0, %1, %2, %3, %4, %5, %6, %7};" ::"r"(p[0]),
        "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]), "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<16>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]), "=r"(p[5]), "=r"(p[6]),
          "=r"(p[7]), "=r"(p[8]), "=r"(p[9]), "=r"(p[10]), "=r"(p[11]), "=r"(p[12]), "=r"(p[13]),
          "=r"(p[14]), "=r"(p[15])
        : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<16>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};" ::
            "r"(p[0]),
        "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]), "r"(p[8]),
        "r"(p[9]), "r"(p[10]), "r"(p[11]), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]),
        "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<32>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]), "=r"(p[5]), "=r"(p[6]),
          "=r"(p[7]), "=r"(p[8]), "=r"(p[9]), "=r"(p[10]), "=r"(p[11]), "=r"(p[12]), "=r"(p[13]),
          "=r"(p[14]), "=r"(p[15]), "=r"(p[16]), "=r"(p[17]), "=r"(p[18]), "=r"(p[19]), "=r"(p[20]),
          "=r"(p[21]), "=r"(p[22]), "=r"(p[23]), "=r"(p[24]), "=r"(p[25]), "=r"(p[26]), "=r"(p[27]),
          "=r"(p[28]), "=r"(p[29]), "=r"(p[30]), "=r"(p[31])
        : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<32>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31};" ::
            "r"(p[0]),
        "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]), "r"(p[8]),
        "r"(p[9]), "r"(p[10]), "r"(p[11]), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]),
        "r"(p[16]), "r"(p[17]), "r"(p[18]), "r"(p[19]), "r"(p[20]), "r"(p[21]), "r"(p[22]),
        "r"(p[23]), "r"(p[24]), "r"(p[25]), "r"(p[26]), "r"(p[27]), "r"(p[28]), "r"(p[29]),
        "r"(p[30]), "r"(p[31]), "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<64>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]), "=r"(p[5]), "=r"(p[6]),
          "=r"(p[7]), "=r"(p[8]), "=r"(p[9]), "=r"(p[10]), "=r"(p[11]), "=r"(p[12]), "=r"(p[13]),
          "=r"(p[14]), "=r"(p[15]), "=r"(p[16]), "=r"(p[17]), "=r"(p[18]), "=r"(p[19]), "=r"(p[20]),
          "=r"(p[21]), "=r"(p[22]), "=r"(p[23]), "=r"(p[24]), "=r"(p[25]), "=r"(p[26]), "=r"(p[27]),
          "=r"(p[28]), "=r"(p[29]), "=r"(p[30]), "=r"(p[31]), "=r"(p[32]), "=r"(p[33]), "=r"(p[34]),
          "=r"(p[35]), "=r"(p[36]), "=r"(p[37]), "=r"(p[38]), "=r"(p[39]), "=r"(p[40]), "=r"(p[41]),
          "=r"(p[42]), "=r"(p[43]), "=r"(p[44]), "=r"(p[45]), "=r"(p[46]), "=r"(p[47]), "=r"(p[48]),
          "=r"(p[49]), "=r"(p[50]), "=r"(p[51]), "=r"(p[52]), "=r"(p[53]), "=r"(p[54]), "=r"(p[55]),
          "=r"(p[56]), "=r"(p[57]), "=r"(p[58]), "=r"(p[59]), "=r"(p[60]), "=r"(p[61]), "=r"(p[62]),
          "=r"(p[63])
        : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<64>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x64.b32 [%64], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63};" ::
            "r"(p[0]),
        "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]), "r"(p[8]),
        "r"(p[9]), "r"(p[10]), "r"(p[11]), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]),
        "r"(p[16]), "r"(p[17]), "r"(p[18]), "r"(p[19]), "r"(p[20]), "r"(p[21]), "r"(p[22]),
        "r"(p[23]), "r"(p[24]), "r"(p[25]), "r"(p[26]), "r"(p[27]), "r"(p[28]), "r"(p[29]),
        "r"(p[30]), "r"(p[31]), "r"(p[32]), "r"(p[33]), "r"(p[34]), "r"(p[35]), "r"(p[36]),
        "r"(p[37]), "r"(p[38]), "r"(p[39]), "r"(p[40]), "r"(p[41]), "r"(p[42]), "r"(p[43]),
        "r"(p[44]), "r"(p[45]), "r"(p[46]), "r"(p[47]), "r"(p[48]), "r"(p[49]), "r"(p[50]),
        "r"(p[51]), "r"(p[52]), "r"(p[53]), "r"(p[54]), "r"(p[55]), "r"(p[56]), "r"(p[57]),
        "r"(p[58]), "r"(p[59]), "r"(p[60]), "r"(p[61]), "r"(p[62]), "r"(p[63]), "r"(addr));
}
template <> DEVICE void tmem_ld_32dp32bNx<128>(uint32_t addr, void* dst) {
    auto p = static_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
        : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]), "=r"(p[5]), "=r"(p[6]),
          "=r"(p[7]), "=r"(p[8]), "=r"(p[9]), "=r"(p[10]), "=r"(p[11]), "=r"(p[12]), "=r"(p[13]),
          "=r"(p[14]), "=r"(p[15]), "=r"(p[16]), "=r"(p[17]), "=r"(p[18]), "=r"(p[19]), "=r"(p[20]),
          "=r"(p[21]), "=r"(p[22]), "=r"(p[23]), "=r"(p[24]), "=r"(p[25]), "=r"(p[26]), "=r"(p[27]),
          "=r"(p[28]), "=r"(p[29]), "=r"(p[30]), "=r"(p[31]), "=r"(p[32]), "=r"(p[33]), "=r"(p[34]),
          "=r"(p[35]), "=r"(p[36]), "=r"(p[37]), "=r"(p[38]), "=r"(p[39]), "=r"(p[40]), "=r"(p[41]),
          "=r"(p[42]), "=r"(p[43]), "=r"(p[44]), "=r"(p[45]), "=r"(p[46]), "=r"(p[47]), "=r"(p[48]),
          "=r"(p[49]), "=r"(p[50]), "=r"(p[51]), "=r"(p[52]), "=r"(p[53]), "=r"(p[54]), "=r"(p[55]),
          "=r"(p[56]), "=r"(p[57]), "=r"(p[58]), "=r"(p[59]), "=r"(p[60]), "=r"(p[61]), "=r"(p[62]),
          "=r"(p[63]), "=r"(p[64]), "=r"(p[65]), "=r"(p[66]), "=r"(p[67]), "=r"(p[68]), "=r"(p[69]),
          "=r"(p[70]), "=r"(p[71]), "=r"(p[72]), "=r"(p[73]), "=r"(p[74]), "=r"(p[75]), "=r"(p[76]),
          "=r"(p[77]), "=r"(p[78]), "=r"(p[79]), "=r"(p[80]), "=r"(p[81]), "=r"(p[82]), "=r"(p[83]),
          "=r"(p[84]), "=r"(p[85]), "=r"(p[86]), "=r"(p[87]), "=r"(p[88]), "=r"(p[89]), "=r"(p[90]),
          "=r"(p[91]), "=r"(p[92]), "=r"(p[93]), "=r"(p[94]), "=r"(p[95]), "=r"(p[96]), "=r"(p[97]),
          "=r"(p[98]), "=r"(p[99]), "=r"(p[100]), "=r"(p[101]), "=r"(p[102]), "=r"(p[103]),
          "=r"(p[104]), "=r"(p[105]), "=r"(p[106]), "=r"(p[107]), "=r"(p[108]), "=r"(p[109]),
          "=r"(p[110]), "=r"(p[111]), "=r"(p[112]), "=r"(p[113]), "=r"(p[114]), "=r"(p[115]),
          "=r"(p[116]), "=r"(p[117]), "=r"(p[118]), "=r"(p[119]), "=r"(p[120]), "=r"(p[121]),
          "=r"(p[122]), "=r"(p[123]), "=r"(p[124]), "=r"(p[125]), "=r"(p[126]), "=r"(p[127])
        : "r"(addr));
}
template <> DEVICE void tmem_st_32dp32bNx<128>(uint32_t addr, const void* src) {
    auto p = static_cast<const uint32_t*>(src);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x128.b32 [%128], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127};" ::
            "r"(p[0]),
        "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]), "r"(p[8]),
        "r"(p[9]), "r"(p[10]), "r"(p[11]), "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]),
        "r"(p[16]), "r"(p[17]), "r"(p[18]), "r"(p[19]), "r"(p[20]), "r"(p[21]), "r"(p[22]),
        "r"(p[23]), "r"(p[24]), "r"(p[25]), "r"(p[26]), "r"(p[27]), "r"(p[28]), "r"(p[29]),
        "r"(p[30]), "r"(p[31]), "r"(p[32]), "r"(p[33]), "r"(p[34]), "r"(p[35]), "r"(p[36]),
        "r"(p[37]), "r"(p[38]), "r"(p[39]), "r"(p[40]), "r"(p[41]), "r"(p[42]), "r"(p[43]),
        "r"(p[44]), "r"(p[45]), "r"(p[46]), "r"(p[47]), "r"(p[48]), "r"(p[49]), "r"(p[50]),
        "r"(p[51]), "r"(p[52]), "r"(p[53]), "r"(p[54]), "r"(p[55]), "r"(p[56]), "r"(p[57]),
        "r"(p[58]), "r"(p[59]), "r"(p[60]), "r"(p[61]), "r"(p[62]), "r"(p[63]), "r"(p[64]),
        "r"(p[65]), "r"(p[66]), "r"(p[67]), "r"(p[68]), "r"(p[69]), "r"(p[70]), "r"(p[71]),
        "r"(p[72]), "r"(p[73]), "r"(p[74]), "r"(p[75]), "r"(p[76]), "r"(p[77]), "r"(p[78]),
        "r"(p[79]), "r"(p[80]), "r"(p[81]), "r"(p[82]), "r"(p[83]), "r"(p[84]), "r"(p[85]),
        "r"(p[86]), "r"(p[87]), "r"(p[88]), "r"(p[89]), "r"(p[90]), "r"(p[91]), "r"(p[92]),
        "r"(p[93]), "r"(p[94]), "r"(p[95]), "r"(p[96]), "r"(p[97]), "r"(p[98]), "r"(p[99]),
        "r"(p[100]), "r"(p[101]), "r"(p[102]), "r"(p[103]), "r"(p[104]), "r"(p[105]), "r"(p[106]),
        "r"(p[107]), "r"(p[108]), "r"(p[109]), "r"(p[110]), "r"(p[111]), "r"(p[112]), "r"(p[113]),
        "r"(p[114]), "r"(p[115]), "r"(p[116]), "r"(p[117]), "r"(p[118]), "r"(p[119]), "r"(p[120]),
        "r"(p[121]), "r"(p[122]), "r"(p[123]), "r"(p[124]), "r"(p[125]), "r"(p[126]), "r"(p[127]),
        "r"(addr));
}
struct SM100_TMEM_STORE_32dp32b1x {
    DEVICE static void copy(uint32_t v, uint32_t addr) { tmem_st_32dp32bNx<1>(addr, &v); }
};
DEVICE void fence_view_async_tmem_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
DEVICE void fence_view_async_tmem_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// SM103 TMEM load with a per-lane maximum reduction.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1100
DEVICE void tmem_ld_32dp32b32x_max(uint32_t tmem_col, float (&p)[32], float& p_max) {
    asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, "
                 "%8, %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, "
                 "%24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                 : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]),
                   "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]),
                   "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]),
                   "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]),
                   "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]),
                   "=f"(p[30]), "=f"(p[31]), "=f"(p_max)
                 : "r"(tmem_col));
}
#endif

} // namespace ptx

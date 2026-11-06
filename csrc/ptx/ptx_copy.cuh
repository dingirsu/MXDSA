#pragma once
namespace ptx {
struct SM90_TMA_STORE_5D
{
  DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);

    asm volatile (
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");

  }
};
struct SM90_BULK_COPY_G2S
{
  DEVICE static void
  copy(void const* gmem_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr, int32_t load_bytes)
  {
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
                     : "memory");

  }

  struct PREFETCH
  {
    DEVICE static void
    copy(void const* gmem_ptr, int32_t load_bytes)
    {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                      :
                      : "l"(gmem_ptr), "r"(load_bytes)
                      : "memory");
  
    }
  };
};
struct SM100_UTCCP_2x64dp128bitlw0123_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
    asm volatile ("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));

  }
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT
{
  DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    // Executed by both CTAs. Set peer bit to 0 so that the
    // transaction bytes will update CTA0's barrier.
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");

  }
};
}

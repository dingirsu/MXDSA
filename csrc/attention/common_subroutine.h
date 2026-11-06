#pragma once
#include "ptx/ptx.cuh"
namespace sm100 {
template<
    int D_V,
    int CHUNK_SIZE,
    int TMEM_COL_START
>
DEVICE
void rescale_O(
    float scale_factor
) {
    float2 scale_factor_float2 = {scale_factor, scale_factor};
    float2 o[CHUNK_SIZE/2];

    UNROLL
    for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
        // Load O
        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        ptx::fence_view_async_tmem_load();

        // Mult
        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_factor_float2);
        }

        // Store O
        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        ptx::fence_view_async_tmem_store();
    }
}

template<int NUM_ELEMS_PER_THREAD>
DEVICE
float get_max(
    float p[NUM_ELEMS_PER_THREAD]
) {
    float local_max = -CUDART_INF_F;
    UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
        local_max = max(local_max, p[i]);
    }
    return local_max;
}


}

namespace sm100 {
DEVICE
char load_indices_and_generate_mask(
    int lane_idx,
    int* gIndices,
    int s_kv,
    int abs_pos_start,
    int topk_length
) {
    int indices[8];
    KU_LDG_256(
        gIndices + lane_idx*8, 
        indices,
        ".nc", 
        "no_allocate", 
        "evict_normal", 
        "256B"
    );
    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
        int abs_pos = abs_pos_start + lane_idx*8 + rel_pos_in_lane;
        return index >= 0 && index < s_kv && abs_pos < topk_length;
    };
    char is_ks_valid_mask = \
        is_valid(7, indices[7]) << 7 | 
        is_valid(6, indices[6]) << 6 | 
        is_valid(5, indices[5]) << 5 |
        is_valid(4, indices[4]) << 4 |
        is_valid(3, indices[3]) << 3 |
        is_valid(2, indices[2]) << 2 |
        is_valid(1, indices[1]) << 1 |
        is_valid(0, indices[0]) << 0;
    return is_ks_valid_mask;
}

}

#pragma once

#include "quantization_utils.cuh"

enum QuantMode {
    MOR = 1,
    BLOCK_SCALING_E4M3 = 2,
    BLOCK_SCALING_E5M2 = 3,
    CURRENT_SCALING_E4M3 = 4,
    CURRENT_SCALING_E5M2 = 5,
    BF16 = 7
};

enum MorDecisionMode {
    DYNAMIC_RANGE = 1,
    QUANT_ERROR = 2
};

enum MorQuantType {
    QUANT_UNKNOWN = 0,
    QUANT_E4M3 = 1,
    QUANT_E5M2 = 2,
    QUANT_BF16 = 3,
    QUANT_E2M3 = 4,
    QUANT_E4M3_SMALL = 11,
    QUANT_E5M2_SMALL = 12
};

__device__ void get_block_amax_amin(const uint32_t* input_device, float* max_sh, float* min_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols);

template<ScalingFactorType T_SF_TYPE>
__device__ void get_block_error(const uint32_t* input_device, float* e4m3_sh, float* e5m2_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float block_amax, float block_amin);

template <ScalingFactorType T_SF_TYPE>
__device__ MorQuantType choose_mor_quant_type(const uint32_t* input_device, float* e4m3_sh, float* e5m2_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float clamp_threshold, MorDecisionMode mor_mode, float block_amax, float block_amin); 
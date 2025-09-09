// Standard C++ libraries.
#include <cassert>
#include <stdio.h>
#include <type_traits>
#include <vector>

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// PyTorch libraries.
#include <torch/extension.h>

#include "cuda_utils.cuh"
#include "quantization_utils.cuh"
#include "warp_reduce.cuh"
#include "mor.cuh"

template <QuantMode T_QUANT_MODE, ScalingFactorType T_SF_TYPE>
__global__ void quantize_kernel(const uint32_t *input_device, uint32_t *output_device, int32_t* meta_device, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float clamp_threshold, MorDecisionMode mor_mode) {
    // Get thread ID.
    extern __shared__ float sharedMem[];

    float* max_sh = &sharedMem[0];
    float* min_sh = &sharedMem[blockDim.x * blockDim.y];
    float block_amax = 0;
    float block_amin = 0;
    if constexpr (T_QUANT_MODE != CURRENT_SCALING_E4M3 and T_QUANT_MODE != CURRENT_SCALING_E5M2) {
	get_block_amax_amin(input_device, max_sh, min_sh, num_rows, num_cols, block_num_rows, block_num_cols);
	block_amax = max_sh[0];
	block_amin = min_sh[0];
    }

    // If the quantization mode is MoR, determine the representation of the block.
    MorQuantType mor_quant_type = QUANT_UNKNOWN;
    if constexpr (T_QUANT_MODE == MOR) {
	float* e4m3_sh = &sharedMem[0];
	float* e5m2_sh = &sharedMem[blockDim.x * blockDim.y];
	mor_quant_type = choose_mor_quant_type<T_SF_TYPE>(input_device, e4m3_sh, e5m2_sh, num_rows, num_cols, block_num_rows, block_num_cols, clamp_threshold, mor_mode, block_amax, block_amin);
	assert(mor_quant_type != QUANT_UNKNOWN);
    }

    // Set metadata.
    int meta_index = blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0 and threadIdx.y == 0) {
	switch (T_QUANT_MODE) {
	    case MOR:
		meta_device[meta_index] = static_cast<int>(mor_quant_type);
		break;
	    case BLOCK_SCALING_E4M3:
		meta_device[meta_index] = 1;
		break;
	    case BLOCK_SCALING_E5M2:
		meta_device[meta_index] = 2;
		break;
	    case CURRENT_SCALING_E4M3:
		meta_device[meta_index] = 1;
		break;
	    case CURRENT_SCALING_E5M2:
		meta_device[meta_index] = 2;
		break;
	    default:
		meta_device[meta_index] = 3;
		break;
	}
    }

    // Quantize the output matrix.
    for (int tid_y = block_num_rows * blockIdx.y + threadIdx.y; tid_y < block_num_rows * (blockIdx.y + 1) && tid_y < num_rows; tid_y += blockDim.y) {
	for (int tid_x = block_num_cols * blockIdx.x + threadIdx.x; tid_x < block_num_cols * (blockIdx.x + 1) &&  tid_x < num_cols ; tid_x += blockDim.x) {
	    int element_id = tid_y * num_cols + tid_x;
	    uint32_t value = input_device[element_id];

	    // Get the two bf16 values;
	    uint32_t bf16_value1 = value >> 16;
	    uint32_t bf16_value2 = value & 0xFFFF;

	    // Convert the bf16 values to fp32. So we will do all the calculations in fp32.
	    float fp32_value1 = bf16_to_fp32(bf16_value1);
	    float fp32_value2 = bf16_to_fp32(bf16_value2);

	    // Default case. Use bf16.
	    uint32_t pruned_bf16_value1 = bf16_value1;
	    uint32_t pruned_bf16_value2 = bf16_value2;

	    float max_normal_fp8_e4m3 = 448;
	    float max_normal_fp8_e5m2 = 57344;
	    float max_normal_e2m3 = 7.5;
	    float scaling_factor = 1.0;
	    switch (T_QUANT_MODE) {
		case MOR:
		    switch (mor_quant_type) {
			case QUANT_E2M3:
			    if (block_amax > 0) {
				scaling_factor = max_normal_e2m3/block_amax;
				assert(scaling_factor < 1e30);
				if constexpr (T_SF_TYPE == SF_E8) {
				    pruned_bf16_value1 = SQDD_e8_e2m3_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_e8_e2m3_bf16(fp32_value2, scaling_factor);
				} else {
				    pruned_bf16_value1 = SQDD_fp32_e2m3_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_fp32_e2m3_bf16(fp32_value2, scaling_factor);
				}
			    }
			    break;
			case QUANT_E4M3:
			case QUANT_E4M3_SMALL:
			    if (block_amax > 0) {
				scaling_factor = max_normal_fp8_e4m3/block_amax;
				assert(scaling_factor < 1e30);
				if constexpr (T_SF_TYPE == SF_E8) {
				    pruned_bf16_value1 = SQDD_e8_e4m3_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_e8_e4m3_bf16(fp32_value2, scaling_factor);
				} else {
				    pruned_bf16_value1 = SQDD_fp32_e4m3_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_fp32_e4m3_bf16(fp32_value2, scaling_factor);
				}
			    }
			    break;
			case QUANT_E5M2:
			case QUANT_E5M2_SMALL:
			    if (block_amax > 0) {
				scaling_factor = max_normal_fp8_e5m2/block_amax;
				assert(scaling_factor < 1e30);
				if constexpr (T_SF_TYPE == SF_E8) {
				    pruned_bf16_value1 = SQDD_e8_e5m2_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_e8_e5m2_bf16(fp32_value2, scaling_factor);
				} else {
				    pruned_bf16_value1 = SQDD_fp32_e5m2_bf16(fp32_value1, scaling_factor);
				    pruned_bf16_value2 = SQDD_fp32_e5m2_bf16(fp32_value2, scaling_factor);
				}
			    }
			    break;
		    }
		    break;
		case BLOCK_SCALING_E4M3:
		    if (block_amax > 0) {
			scaling_factor = max_normal_fp8_e4m3/block_amax;
			if constexpr (T_SF_TYPE == SF_E8) {
			    pruned_bf16_value1 = SQDD_e8_e4m3_bf16(fp32_value1, scaling_factor);
			    pruned_bf16_value2 = SQDD_e8_e4m3_bf16(fp32_value2, scaling_factor);
			} else {
			    pruned_bf16_value1 = SQDD_fp32_e4m3_bf16(fp32_value1, scaling_factor);
			    pruned_bf16_value2 = SQDD_fp32_e4m3_bf16(fp32_value2, scaling_factor);
			}
		    }
		    break;
		case BLOCK_SCALING_E5M2:
		    if (block_amax > 0) {
			scaling_factor = max_normal_fp8_e5m2/block_amax;
			if constexpr (T_SF_TYPE == SF_E8) {
			    pruned_bf16_value1 = SQDD_e8_e5m2_bf16(fp32_value1, scaling_factor);
			    pruned_bf16_value2 = SQDD_e8_e5m2_bf16(fp32_value2, scaling_factor);
			} else {
			    pruned_bf16_value1 = SQDD_fp32_e5m2_bf16(fp32_value1, scaling_factor);
			    pruned_bf16_value2 = SQDD_fp32_e5m2_bf16(fp32_value2, scaling_factor);
			}
		    }
		    break;
		case CURRENT_SCALING_E4M3:
		    if constexpr (T_SF_TYPE == SF_E8) {
			pruned_bf16_value1 = SQDD_e8_e4m3_bf16(fp32_value1, clamp_threshold);
			pruned_bf16_value2 = SQDD_e8_e4m3_bf16(fp32_value2, clamp_threshold);
		    } else {
			pruned_bf16_value1 = SQDD_fp32_e4m3_bf16(fp32_value1, clamp_threshold);
			pruned_bf16_value2 = SQDD_fp32_e4m3_bf16(fp32_value2, clamp_threshold);
		    }
		    break;
		case CURRENT_SCALING_E5M2:
		    if constexpr (T_SF_TYPE == SF_E8) {
			pruned_bf16_value1 = SQDD_e8_e5m2_bf16(fp32_value1, clamp_threshold);
			pruned_bf16_value2 = SQDD_e8_e5m2_bf16(fp32_value2, clamp_threshold);
		    } else {
			pruned_bf16_value1 = SQDD_fp32_e5m2_bf16(fp32_value1, clamp_threshold);
			pruned_bf16_value2 = SQDD_fp32_e5m2_bf16(fp32_value2, clamp_threshold);
		    }
		    break;
		default:
		    break;
	    }

	    // Merge the two bf16 values to one uint32 value.
	    uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;
	    output_device[element_id] = merged_bf16;
	}
    }
}

template <int T_NUM_EXPONENTS, int T_NUM_MANTISSAS, bool T_USE_CUDA_CAST>
__global__ void quant_dequant_kernel(const float *input_device, float *output_device, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols) {
    for (int tid_y = block_num_rows * blockIdx.y + threadIdx.y; tid_y < block_num_rows * (blockIdx.y + 1) && tid_y < num_rows; tid_y += blockDim.y) {
	for (int tid_x = block_num_cols * blockIdx.x + threadIdx.x; tid_x < block_num_cols * (blockIdx.x + 1) &&  tid_x < num_cols ; tid_x += blockDim.x) {
	    int element_id = tid_y * num_cols + tid_x;
	    float value = input_device[element_id];
	    output_device[element_id] = quant_dequant_common<T_NUM_EXPONENTS, T_NUM_MANTISSAS, T_USE_CUDA_CAST>(value);
	}
    }
}

template <E8ScalingRoundingType T_E8_SCALING_ROUNDING>
__global__ void e4m3_with_e8_scale_kernel(const float *input_device, float *output_device, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols) {
    for (int tid_y = block_num_rows * blockIdx.y + threadIdx.y; tid_y < block_num_rows * (blockIdx.y + 1) && tid_y < num_rows; tid_y += blockDim.y) {
	for (int tid_x = block_num_cols * blockIdx.x + threadIdx.x; tid_x < block_num_cols * (blockIdx.x + 1) &&  tid_x < num_cols ; tid_x += blockDim.x) {
	    int element_id = tid_y * num_cols + tid_x;
	    float value = input_device[element_id];
	    float scaling_factor = 448.0 / value;
	    output_device[element_id] = SQDD_e8_scale<T_E8_SCALING_ROUNDING, 4, 3, false>(value, scaling_factor);
	}
    }
}



__global__ void block_amax_and_non_zero(
    const uint32_t* input_device,
    float* block_amax_device,
    int* block_non_zero_device,
    int num_rows,
    int num_cols,
    int block_num_rows,
    int block_num_cols) {
    int threads_per_block = blockDim.x * blockDim.y;
    int warps_per_block = threads_per_block / warpSize;

    extern __shared__ float sharedMem[];

    float* s_amax = sharedMem;
    int* s_non_zero = (int*)((char*)sharedMem + (warps_per_block * sizeof(float)));

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 1: Compute local block maxima
    //////////////////////////////////////////////////////////////////////////////////
    int globalBlockRow = block_num_rows * blockIdx.y;
    int globalBlockCol = block_num_cols * blockIdx.x;

    float local_amax = -INFINITY;
    int local_non_zero = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    for (int r = 0; r < block_num_rows; r += blockDim.y)
    {
	int row = globalBlockRow + r + threadIdx.y;

	for (int c = 0; c < block_num_cols; c += blockDim.x)
	{
	    int col = globalBlockCol + c + threadIdx.x;

	    uint32_t value = input_device[row * num_cols + col];

	    uint32_t bf16_value1 = value >> 16;
	    uint32_t bf16_value2 = value & 0xFFFF;

	    float fp32_value1 = bf16_to_fp32(bf16_value1);
	    float fp32_value2 = bf16_to_fp32(bf16_value2);

	    float abs_value1 = fabsf(fp32_value1);
	    float abs_value2 = fabsf(fp32_value2);

	    if (abs_value1 > 0) local_non_zero++;
	    if (abs_value2 > 0) local_non_zero++;

	    local_amax = fmaxf(local_amax, abs_value1);
	    local_amax = fmaxf(local_amax, abs_value2);
	}
    }

    // Reduce within the block
    blockReduceAmaxNonZero(local_amax, local_non_zero, s_amax, s_non_zero, tid);

    // Write results to global memory
    if (tid == 0) 
    {
	block_amax_device[block_id] = s_amax[0];
	block_non_zero_device[block_id] = s_non_zero[0];
    }
}

__global__ void global_amax_and_non_zero(
    float* block_amax_device,
    int* block_non_zero_device,
    float* global_amax_device,
    int* global_non_zero_device,
    int num_blocks) {

    float global_amax = -FLT_MAX;
    int global_non_zero = 0;

    int tid = threadIdx.x;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        global_amax = fmaxf(global_amax, block_amax_device[i]);
	global_non_zero += block_non_zero_device[i];
    }
    
    int threads_per_block = blockDim.x;
    int warps_per_block = threads_per_block / warpSize;
    extern __shared__ float sharedMem[];
    float* s_amax = sharedMem;
    int* s_non_zero = (int*)((char*)sharedMem + (warps_per_block * sizeof(float)));

    blockReduceAmaxNonZero(global_amax, global_non_zero, s_amax, s_non_zero, tid);

    global_amax = s_amax[0];
    global_non_zero = s_non_zero[0];
    
    if (tid == 0) {
	global_amax_device[0] = global_amax;
	global_non_zero_device[0] = global_non_zero;
	assert(global_non_zero > 0);
    }
}

enum ScalingStrategy {
    SS_GLOBAL_AMAX = 1,
    SS_BLOCK_AMAX = 2,
    SS_E8 = 3,
    SS_CURRENT_SCALING = 4,
};


template <ScalingStrategy T_SCALING_STRATEGY>
__device__ float get_scaling_factor(float global_amax, float block_amax) {
    float scaling_factor = 1.0; //initialize it to something...
    float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
    float epsilon = 1e-35;  // The smallest allowed block amax.
    if (block_amax < epsilon) block_amax = epsilon;
    switch (T_SCALING_STRATEGY) {
	case SS_GLOBAL_AMAX:
	    float ratio = max_fp8_e4m3 / global_amax;
	    uint32_t matrix_sf_mantissa = *reinterpret_cast<uint32_t*>(&ratio) & 0x007FFFFF;
	    if (block_amax > 0) {
	        float block_sf = max_fp8_e4m3 / block_amax;
	        uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
	        uint32_t block_sf_mantissa = block_sf_bits & 0x007FFFFF;
	        uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
	        float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
	        scaling_factor = (matrix_sf_mantissa <= block_sf_mantissa) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;
	    }
	    break;
	case SS_BLOCK_AMAX:
	case SS_E8:
	    if (block_amax > 0) {
	        scaling_factor = max_fp8_e4m3 / block_amax;
	    }
	    break;
	case SS_CURRENT_SCALING:
	    if (global_amax > 0) {
	        scaling_factor = max_fp8_e4m3 / global_amax;
	    }
	    break;
	default:
	    break;
    }
    return scaling_factor;
}

template <ScalingStrategy T_SCALING_STRATEGY>
__global__ void block_e4m3_relative_error(
    const uint32_t* input_device,
    uint32_t* output_device,
    float* block_amax_device,
    float* block_error_device,
    float* global_amax_device,
    int num_rows,
    int num_cols,
    int block_num_rows,
    int block_num_cols) {
    float global_amax = global_amax_device[0];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    float block_amax = block_amax_device[block_id];
    float scaling_factor = get_scaling_factor<T_SCALING_STRATEGY>(global_amax, block_amax);

    // Quantize values in the sub-tensor
    float local_error = 0.0;
    int globalBlockRow = block_num_rows * blockIdx.y;
    int globalBlockCol = block_num_cols * blockIdx.x;

    for (int r = 0; r < block_num_rows; r += blockDim.y)
    {
	int row = globalBlockRow + r + threadIdx.y;

	for (int c = 0; c < block_num_cols; c += blockDim.x)
	{
	    int col = globalBlockCol + c + threadIdx.x;

	    // Load value from global memory
	    uint32_t value = input_device[row * num_cols + col];
	    uint32_t bf16_value1 = value >> 16;
	    uint32_t bf16_value2 = value & 0xFFFF;

	    float fp32_value1 = bf16_to_fp32(bf16_value1);
	    float fp32_value2 = bf16_to_fp32(bf16_value2);

	    float dequantized_value1 = 0.0f;
	    float dequantized_value2 = 0.0f;

	    if (fp32_value1 != 0)
	    {
		if constexpr (T_SCALING_STRATEGY == SF_E8) {
		    dequantized_value1 = SQDD_e8_e4m3_fp32(fp32_value1, scaling_factor);
		} else {
		    dequantized_value1 = SQDD_fp32_e4m3_fp32(fp32_value1, scaling_factor);
		}
		float value_error = fabsf((dequantized_value1 - fp32_value1) / fp32_value1);
		local_error += value_error;
	    }

	    if (fp32_value2 != 0)
	    {
		if constexpr (T_SCALING_STRATEGY == SF_E8) {
		    dequantized_value2 = SQDD_e8_e4m3_fp32(fp32_value2, scaling_factor);
		} else {
		    dequantized_value2 = SQDD_fp32_e4m3_fp32(fp32_value2, scaling_factor);
		}
		float value_error = fabsf((dequantized_value2 - fp32_value2) / fp32_value2);
		local_error += value_error;
	    }

	    uint32_t pruned_bf16_value1 = fp32_to_bf16(dequantized_value1);
	    uint32_t pruned_bf16_value2 = fp32_to_bf16(dequantized_value2);
	    uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;

	    output_device[row * num_cols + col] = merged_bf16;
	}
    }

    // Reduce within the block
    extern __shared__ float sharedMem[];
    float* s_amax = sharedMem;
    blockReduceError(local_error, s_amax, tid);

    // Write results to global memory
    if (tid == 0) 
    {
	block_error_device[block_id] = s_amax[0];
    }
}

__global__ void global_e4m3_relative_error(
    float* block_error_device,
    float* global_error_device,
    int num_blocks) {

    float global_error = 0;

    int tid = threadIdx.x;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
	global_error += block_error_device[i];
    }
    
    int threads_per_block = blockDim.x;
    int warps_per_block = threads_per_block / warpSize;
    extern __shared__ float sharedMem[];
    float* s_amax = sharedMem;

    blockReduceError(global_error, s_amax, tid);
    global_error = s_amax[0];
    
    if (tid == 0) {
	global_error_device[0] = global_error;
    }
}

__global__ void maybe_use_bf16(
    const uint32_t* input_device,
    uint32_t* output_device,
    int* global_non_zero_device,
    float* global_error_device,
    int* meta_device,
    int num_rows,
    int num_cols,
    int block_num_rows,
    int block_num_cols,
    float e4m3_threshold) {
    float average_relative_error = global_error_device[0]/global_non_zero_device[0];
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (average_relative_error > e4m3_threshold) {
	if (block_id == 0 && tid == 0) {
	    meta_device[0] = 3;
	}

	//We are going to store in BF16, and basically copy from A to B :)
	int globalBlockRow = block_num_rows * blockIdx.y;
	int globalBlockCol = block_num_cols * blockIdx.x;

	for (int r = 0; r < block_num_rows; r += blockDim.y)
	{
	    int row = globalBlockRow + r + threadIdx.y;

	    for (int c = 0; c < block_num_cols; c += blockDim.x)
	    {
		int col = globalBlockCol + c + threadIdx.x;
		output_device[row * num_cols + col] = input_device[row * num_cols + col];
	    }
	}
    } else {
	if (block_id == 0 && tid == 0) {
	    meta_device[0] = 1;
	}
    }
}

namespace mor {

dim3 get_grid_size(int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    int uint32_num_cols = num_cols / 2;
    int uint32_block_num_cols = block_num_cols / 2;
    return dim3((uint32_num_cols - 1) / uint32_block_num_cols + 1, (num_rows - 1) / block_num_rows + 1);
}

void call_quantize_kernel(uint32_t* input_uint32_ptr, uint32_t* result_uint32_ptr, int32_t* meta_ptr, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double clamp_threshold, int64_t mode, int64_t sf_type) {
    // Calculate blocksize and gridsize.
    // Dimension x is the contiguous dimension. So it is the column ID.
    // Dimension y is the non-contiguous dimension. So it is the row ID.
    TORCH_CHECK(num_cols % 2 == 0);
    int uint32_num_cols = num_cols / 2;
    int uint32_block_num_cols = block_num_cols / 2;
    int tb_block_x = 16;
    int tb_block_y = 32;

    // Retrieve the quantization mode from the mode variable.
    QuantMode quant_mode = static_cast<QuantMode>(mode % 10);
    // Retrieve the mor decision mode from the mode variable.
    MorDecisionMode mor_mode = QUANT_ERROR;
    if (mode / 100 > 0) {
	mor_mode = DYNAMIC_RANGE;
    }

    if (block_num_rows * block_num_cols > 1024) {
        assert(block_num_cols >= 32);
        assert(block_num_rows >= 32);
    } else {
        tb_block_x = uint32_block_num_cols;
        tb_block_y = block_num_rows;
    }

    dim3 blockSize(tb_block_x, tb_block_y, 1);
    dim3 gridSize = get_grid_size(num_rows, num_cols, block_num_rows, block_num_cols);

    ScalingFactorType scaling_factor_type = static_cast<ScalingFactorType>(sf_type);

    // Launch CUDA kernel.
    switch (quant_mode) {
	case MOR:
	    if (scaling_factor_type == SF_E8) {
		quantize_kernel<MOR, SF_E8><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold, mor_mode);
	    } else {
		quantize_kernel<MOR, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold, mor_mode);
	    }
	    break;
	case BLOCK_SCALING_E4M3:
	    if (scaling_factor_type == SF_E8) {
		quantize_kernel<BLOCK_SCALING_E4M3, SF_E8><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    } else {
		quantize_kernel<BLOCK_SCALING_E4M3, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    }
	    break;
	case BLOCK_SCALING_E5M2:
	    if (scaling_factor_type == SF_E8) {
		quantize_kernel<BLOCK_SCALING_E5M2, SF_E8><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    } else {
		quantize_kernel<BLOCK_SCALING_E5M2, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    }
	    break;
	case CURRENT_SCALING_E4M3:
	    if (scaling_factor_type == SF_E8) {
		quantize_kernel<CURRENT_SCALING_E4M3, SF_E8><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    } else {
		quantize_kernel<CURRENT_SCALING_E4M3, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    }
	    break;
	case CURRENT_SCALING_E5M2:
	    if (scaling_factor_type == SF_E8) {
		quantize_kernel<CURRENT_SCALING_E5M2, SF_E8><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    } else {
		quantize_kernel<CURRENT_SCALING_E5M2, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    }
	    break;
	default:
	    quantize_kernel<BF16, SF_FP32><<<gridSize, blockSize, tb_block_x * tb_block_y * sizeof(float) * 2>>>(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, uint32_num_cols, block_num_rows, uint32_block_num_cols, clamp_threshold,  mor_mode);
	    break;
    }
}

// Given an input 2D tensor with bf16 type, we will use MoR to quantize each block, and then cast back to bf16.
// We assume that the input tensor is row major, so the elements in one row are contiguous.
std::vector<at::Tensor> fake_quantize_cuda(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double clamp_threshold, int64_t mode, int64_t sf_type) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kBFloat16);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<at::BFloat16>();
    uint32_t* input_uint32_ptr = reinterpret_cast<uint32_t*>(input_ptr);

    std::vector<at::Tensor> output(2);

    // Create the tensor for the quantized matrix.
    output[0] = torch::empty(input_contig.sizes(), input_contig.options());
    TORCH_CHECK(output[0].dtype() == at::kBFloat16);
    TORCH_CHECK(output[0].is_contiguous());
    TORCH_INTERNAL_ASSERT(output[0].device().type() == at::DeviceType::CUDA);
    auto result_ptr = output[0].data_ptr<at::BFloat16>();
    uint32_t* result_uint32_ptr = reinterpret_cast<uint32_t*>(result_ptr);

    // Create the metadata matrix.
    auto meta_options =
	torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    dim3 gridSize = get_grid_size(num_rows, num_cols, block_num_rows, block_num_cols);
    output[1] = torch::empty({gridSize.y, gridSize.x}, meta_options);
    TORCH_CHECK(output[1].is_contiguous());
    TORCH_INTERNAL_ASSERT(output[1].device().type() == at::DeviceType::CUDA);
    int32_t* meta_ptr = output[1].data_ptr<int32_t>();
    
    call_quantize_kernel(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, num_cols, block_num_rows, block_num_cols, clamp_threshold, mode, sf_type);

    return output;
}


// Given an input 2D tensor with bf16 type, we will use MoR to quantize each block, and then cast back to bf16.
// We assume that the input tensor is row major, so the elements in one row are contiguous.
at::Tensor fake_quantize_cuda_no_alloc(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double clamp_threshold, int64_t mode, int64_t sf_type, at::Tensor& result) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kBFloat16);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<at::BFloat16>();
    uint32_t* input_uint32_ptr = reinterpret_cast<uint32_t*>(input_ptr);

    // Check the properties of the result matrix.
    TORCH_CHECK(result.dtype() == at::kBFloat16);
    TORCH_CHECK(result.is_contiguous());
    TORCH_CHECK(result.sizes()[0] == num_rows);
    TORCH_CHECK(result.sizes()[1] == num_cols);
    TORCH_INTERNAL_ASSERT(result.device().type() == at::DeviceType::CUDA);
    auto result_ptr = result.data_ptr<at::BFloat16>();
    uint32_t* result_uint32_ptr = reinterpret_cast<uint32_t*>(result_ptr);

    // Create the metadata matrix.
    auto meta_options =
	torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    dim3 gridSize = get_grid_size(num_rows, num_cols, block_num_rows, block_num_cols);
    auto meta_tensor = torch::empty({gridSize.y, gridSize.x}, meta_options);
    TORCH_CHECK(meta_tensor.is_contiguous());
    TORCH_INTERNAL_ASSERT(meta_tensor.device().type() == at::DeviceType::CUDA);
    int32_t* meta_ptr = meta_tensor.data_ptr<int32_t>();

    call_quantize_kernel(input_uint32_ptr, result_uint32_ptr, meta_ptr, num_rows, num_cols, block_num_rows, block_num_cols, clamp_threshold, mode, sf_type);

    return meta_tensor;
}

template <int T_NUM_EXPONENTS, int T_NUM_MANTISSAS>
at::Tensor quant_dequant(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kFloat);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<float>();

    // Create the tensor for the quantized matrix.
    auto output_tensor = torch::empty(input_contig.sizes(), input_contig.options());
    TORCH_CHECK(output_tensor.dtype() == at::kFloat);
    TORCH_CHECK(output_tensor.is_contiguous());
    TORCH_INTERNAL_ASSERT(output_tensor.device().type() == at::DeviceType::CUDA);
    auto result_ptr = output_tensor.data_ptr<float>();

    // Calculate blocksize and gridsize.
    // Dimension x is the contiguous dimension. So it is the column ID.
    // Dimension y is the non-contiguous dimension. So it is the row ID.
    int tb_block_x = 32;
    int tb_block_y = 32;

    if (block_num_rows * block_num_cols > 1024) {
        assert(block_num_cols >= 32);
        assert(block_num_rows >= 32);
    } else {
        tb_block_x = block_num_cols;
        tb_block_y = block_num_rows;
    }

    dim3 blockSize(tb_block_x, tb_block_y, 1);
    dim3 gridSize((num_cols - 1) / block_num_cols + 1, (num_rows - 1) / block_num_rows + 1);

    // Launch CUDA kernel.
    // quant_dequant_kernel<T_NUM_EXPONENTS, T_NUM_MANTISSAS, true><<<gridSize, blockSize>>>(input_ptr, result_ptr, num_rows, num_cols, block_num_rows, block_num_cols);
    quant_dequant_kernel<T_NUM_EXPONENTS, T_NUM_MANTISSAS, false><<<gridSize, blockSize>>>(input_ptr, result_ptr, num_rows, num_cols, block_num_rows, block_num_cols);

    return output_tensor;
}

at::Tensor quant_dequant_e4m3(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return quant_dequant<4, 3>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

at::Tensor quant_dequant_e5m2(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return quant_dequant<5, 2>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

at::Tensor quant_dequant_e3m2(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return quant_dequant<3, 2>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

at::Tensor quant_dequant_e2m3(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return quant_dequant<2, 3>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

template <E8ScalingRoundingType T_E8_SCALING_ROUNDING>
at::Tensor e4m3_with_e8_scale(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kFloat);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<float>();

    // Create the tensor for the quantized matrix.
    auto output_tensor = torch::empty(input_contig.sizes(), input_contig.options());
    TORCH_CHECK(output_tensor.dtype() == at::kFloat);
    TORCH_CHECK(output_tensor.is_contiguous());
    TORCH_INTERNAL_ASSERT(output_tensor.device().type() == at::DeviceType::CUDA);
    auto result_ptr = output_tensor.data_ptr<float>();

    // Calculate blocksize and gridsize.
    // Dimension x is the contiguous dimension. So it is the column ID.
    // Dimension y is the non-contiguous dimension. So it is the row ID.
    int tb_block_x = 32;
    int tb_block_y = 32;

    if (block_num_rows * block_num_cols > 1024) {
        assert(block_num_cols >= 32);
        assert(block_num_rows >= 32);
    } else {
        tb_block_x = block_num_cols;
        tb_block_y = block_num_rows;
    }

    dim3 blockSize(tb_block_x, tb_block_y, 1);
    dim3 gridSize((num_cols - 1) / block_num_cols + 1, (num_rows - 1) / block_num_rows + 1);

    e4m3_with_e8_scale_kernel<T_E8_SCALING_ROUNDING><<<gridSize, blockSize>>>(input_ptr, result_ptr, num_rows, num_cols, block_num_rows, block_num_cols);

    return output_tensor;
}

at::Tensor e4m3_with_e8_scale_rne(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return e4m3_with_e8_scale<E8_RNE>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

at::Tensor e4m3_with_e8_scale_rz(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols) {
    return e4m3_with_e8_scale<E8_RZ>(input, num_rows, num_cols, block_num_rows, block_num_cols);
}

template <ScalingStrategy T_SCALING_STRATEGY>
std::vector<at::Tensor> fake_quantize_block_scaling_template(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kBFloat16);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_CHECK((num_rows % block_num_rows) == 0);
    TORCH_CHECK((num_cols % block_num_cols) == 0);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<at::BFloat16>();
    uint32_t* input_uint32_ptr = reinterpret_cast<uint32_t*>(input_ptr);


    std::vector<at::Tensor> output(5);

    // Create the tensor for the quantized matrix.
    output[0] = torch::empty(input_contig.sizes(), input_contig.options());
    TORCH_CHECK(output[0].dtype() == at::kBFloat16);
    TORCH_CHECK(output[0].is_contiguous());
    TORCH_INTERNAL_ASSERT(output[0].device().type() == at::DeviceType::CUDA);
    auto result_ptr = output[0].data_ptr<at::BFloat16>();
    uint32_t* result_uint32_ptr = reinterpret_cast<uint32_t*>(result_ptr);

    auto int_options =
	torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    auto float_options =
	torch::TensorOptions().dtype(torch::kFloat).device(input.device());
    // Create the metadata scalar.
    output[1] = torch::empty({1}, int_options);
    TORCH_INTERNAL_ASSERT(output[1].device().type() == at::DeviceType::CUDA);
    int32_t* meta_ptr = output[1].data_ptr<int32_t>();
    // Create the global amax scalar.
    output[2] = torch::empty({1}, float_options);
    TORCH_INTERNAL_ASSERT(output[2].device().type() == at::DeviceType::CUDA);
    float* global_amax_ptr = output[2].data_ptr<float>();
    // Create the global non zero scalar.
    output[3] = torch::empty({1}, int_options);
    TORCH_INTERNAL_ASSERT(output[3].device().type() == at::DeviceType::CUDA);
    int32_t* global_non_zero_ptr = output[3].data_ptr<int32_t>();
    // Create the global e4m3 error scalar.
    output[4] = torch::empty({1}, float_options);
    TORCH_INTERNAL_ASSERT(output[4].device().type() == at::DeviceType::CUDA);
    float* global_error_ptr = output[4].data_ptr<float>();

    int row_packed = num_rows;
    int col_packed = num_cols / 2; //We are packing 2 bf16's in 1 fp32

    int block_num_rows_packed = block_num_rows;
    int block_num_cols_packed = block_num_cols / 2;

    int row_num_blocks = row_packed / block_num_rows_packed;
    int col_num_blocks = col_packed / block_num_cols_packed;
    int num_blocks = col_num_blocks * row_num_blocks;


    // Create temporary tensors.
    auto block_amax = torch::empty({num_blocks}, float_options);
    TORCH_INTERNAL_ASSERT(block_amax.device().type() == at::DeviceType::CUDA);
    float* block_amax_ptr = block_amax.data_ptr<float>();
    auto block_non_zero = torch::empty({num_blocks}, int_options);
    TORCH_INTERNAL_ASSERT(block_non_zero.device().type() == at::DeviceType::CUDA);
    int* block_non_zero_ptr = block_non_zero.data_ptr<int32_t>();
    auto block_error = torch::empty({num_blocks}, float_options);
    TORCH_INTERNAL_ASSERT(block_error.device().type() == at::DeviceType::CUDA);
    float* block_error_ptr = block_error.data_ptr<float>();

    int threads_x = 32;
    int threads_y = 32;

    int threads_per_block = threads_x * threads_y;
    TORCH_CHECK((threads_per_block % 32) == 0);
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim = get_grid_size(num_rows, num_cols, block_num_rows, block_num_cols);  // Ensure grid fits total sub-tensors
    dim3 reductionBlockDim(1024, 1);  // 256 threads per block
    dim3 reductionGridDim(1, 1);
    int warps_per_block = threads_per_block / 32;

    int sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    float e4m3_threshold_fp32 = static_cast<float>(e4m3_threshold);
    block_amax_and_non_zero<<<gridDim, blockDim, sharedMemSize>>>(input_uint32_ptr, block_amax_ptr, block_non_zero_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed);
    global_amax_and_non_zero<<<reductionGridDim, reductionBlockDim, sharedMemSize>>>(block_amax_ptr, block_non_zero_ptr, global_amax_ptr, global_non_zero_ptr, num_blocks);
    block_e4m3_relative_error<T_SCALING_STRATEGY><<<gridDim, blockDim, sharedMemSize>>>(input_uint32_ptr, result_uint32_ptr, block_amax_ptr, block_error_ptr, global_amax_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed);
    // cudaDeviceSynchronize();
    global_e4m3_relative_error<<<reductionGridDim, reductionBlockDim, sharedMemSize>>>(block_error_ptr, global_error_ptr, num_blocks);
    maybe_use_bf16<<<gridDim, blockDim>>>(input_uint32_ptr, result_uint32_ptr, global_non_zero_ptr, global_error_ptr, meta_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed, e4m3_threshold_fp32);

    return output;
}

std::vector<at::Tensor> fake_quantize_block_scaling_gmax(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return fake_quantize_block_scaling_template<SS_GLOBAL_AMAX>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> fake_quantize_block_scaling_bmax(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return fake_quantize_block_scaling_template<SS_BLOCK_AMAX>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> fake_quantize_block_scaling_e8(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return fake_quantize_block_scaling_template<SS_E8>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> fake_quantize_current_scaling(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return fake_quantize_block_scaling_template<SS_CURRENT_SCALING>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> fake_quantize_block_scaling(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold, int64_t ss_mode) {
    ScalingStrategy scaling_strategy = static_cast<ScalingStrategy>(ss_mode);
    switch (scaling_strategy) {
	case SS_GLOBAL_AMAX:
	    return fake_quantize_block_scaling_gmax(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_BLOCK_AMAX:
	    return fake_quantize_block_scaling_bmax(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_E8:
	    return fake_quantize_block_scaling_e8(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_CURRENT_SCALING:
	    return fake_quantize_current_scaling(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
    }	
}

template <ScalingStrategy T_SCALING_STRATEGY>
std::vector<at::Tensor> inplace_fake_quantize_block_scaling_template(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    // Sanity checks for the input 2D tensor.
    TORCH_CHECK(input.dtype() == at::kBFloat16);
    TORCH_CHECK(input.sizes()[0] == num_rows);
    TORCH_CHECK(input.sizes()[1] == num_cols);
    TORCH_CHECK((num_rows % block_num_rows) == 0);
    TORCH_CHECK((num_cols % block_num_cols) == 0);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
    at::Tensor input_contig = input.contiguous();
    auto input_ptr = input_contig.data_ptr<at::BFloat16>();
    uint32_t* input_uint32_ptr = reinterpret_cast<uint32_t*>(input_ptr);


    std::vector<at::Tensor> output(4);

    auto int_options =
	torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    auto float_options =
	torch::TensorOptions().dtype(torch::kFloat).device(input.device());
    // Create the metadata scalar.
    output[0] = torch::empty({1}, int_options);
    TORCH_INTERNAL_ASSERT(output[0].device().type() == at::DeviceType::CUDA);
    int32_t* meta_ptr = output[0].data_ptr<int32_t>();
    // Create the global amax scalar.
    output[1] = torch::empty({1}, float_options);
    TORCH_INTERNAL_ASSERT(output[1].device().type() == at::DeviceType::CUDA);
    float* global_amax_ptr = output[1].data_ptr<float>();
    // Create the global non zero scalar.
    output[2] = torch::empty({1}, int_options);
    TORCH_INTERNAL_ASSERT(output[2].device().type() == at::DeviceType::CUDA);
    int32_t* global_non_zero_ptr = output[2].data_ptr<int32_t>();
    // Create the global e4m3 error scalar.
    output[3] = torch::empty({1}, float_options);
    TORCH_INTERNAL_ASSERT(output[3].device().type() == at::DeviceType::CUDA);
    float* global_error_ptr = output[3].data_ptr<float>();

    int row_packed = num_rows;
    int col_packed = num_cols / 2; //We are packing 2 bf16's in 1 fp32

    int block_num_rows_packed = block_num_rows;
    int block_num_cols_packed = block_num_cols / 2;

    int row_num_blocks = row_packed / block_num_rows_packed;
    int col_num_blocks = col_packed / block_num_cols_packed;
    int num_blocks = col_num_blocks * row_num_blocks;

    // Create temporary tensors.
    auto block_amax = torch::empty({num_blocks}, float_options);
    TORCH_INTERNAL_ASSERT(block_amax.device().type() == at::DeviceType::CUDA);
    float* block_amax_ptr = block_amax.data_ptr<float>();
    auto block_non_zero = torch::empty({num_blocks}, int_options);
    TORCH_INTERNAL_ASSERT(block_non_zero.device().type() == at::DeviceType::CUDA);
    int* block_non_zero_ptr = block_non_zero.data_ptr<int32_t>();
    auto block_error = torch::empty({num_blocks}, float_options);
    TORCH_INTERNAL_ASSERT(block_error.device().type() == at::DeviceType::CUDA);
    float* block_error_ptr = block_error.data_ptr<float>();
    auto result = torch::empty(input_contig.sizes(), input_contig.options());
    TORCH_CHECK(result.dtype() == at::kBFloat16);
    TORCH_CHECK(result.is_contiguous());
    TORCH_INTERNAL_ASSERT(result.device().type() == at::DeviceType::CUDA);
    auto result_ptr = result.data_ptr<at::BFloat16>();
    uint32_t* result_uint32_ptr = reinterpret_cast<uint32_t*>(result_ptr);

    int threads_x = 32;
    int threads_y = 32;

    int threads_per_block = threads_x * threads_y;
    TORCH_CHECK((threads_per_block % 32) == 0);
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim = get_grid_size(num_rows, num_cols, block_num_rows, block_num_cols);  // Ensure grid fits total sub-tensors
    dim3 reductionBlockDim(1024, 1);  // 256 threads per block
    dim3 reductionGridDim(1, 1);

    int warps_per_block = threads_per_block / 32;

    int sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    float e4m3_threshold_fp32 = static_cast<float>(e4m3_threshold);

    block_amax_and_non_zero<<<gridDim, blockDim, sharedMemSize>>>(input_uint32_ptr, block_amax_ptr, block_non_zero_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed);
    global_amax_and_non_zero<<<reductionGridDim, reductionBlockDim, sharedMemSize>>>(block_amax_ptr, block_non_zero_ptr, global_amax_ptr, global_non_zero_ptr, num_blocks);
    block_e4m3_relative_error<T_SCALING_STRATEGY><<<gridDim, blockDim, sharedMemSize>>>(input_uint32_ptr, result_uint32_ptr, block_amax_ptr, block_error_ptr, global_amax_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed);
    // cudaDeviceSynchronize();
    global_e4m3_relative_error<<<reductionGridDim, reductionBlockDim, sharedMemSize>>>(block_error_ptr, global_error_ptr, num_blocks);
    maybe_use_bf16<<<gridDim, blockDim>>>(input_uint32_ptr, result_uint32_ptr, global_non_zero_ptr, global_error_ptr, meta_ptr, row_packed, col_packed, block_num_rows_packed, block_num_cols_packed, e4m3_threshold_fp32);

    CHECK_CUDA_ERROR(cudaMemcpy(input_uint32_ptr, result_uint32_ptr, row_packed * col_packed * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    return output;
}

std::vector<at::Tensor> inplace_fake_quantize_block_scaling_gmax(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return inplace_fake_quantize_block_scaling_template<SS_GLOBAL_AMAX>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> inplace_fake_quantize_block_scaling_bmax(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return inplace_fake_quantize_block_scaling_template<SS_BLOCK_AMAX>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> inplace_fake_quantize_block_scaling_e8(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return inplace_fake_quantize_block_scaling_template<SS_E8>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> inplace_fake_quantize_current_scaling(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
    return inplace_fake_quantize_block_scaling_template<SS_CURRENT_SCALING>(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
}

std::vector<at::Tensor> inplace_fake_quantize_block_scaling(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold, int64_t ss_mode) {
    ScalingStrategy scaling_strategy = static_cast<ScalingStrategy>(ss_mode);
    switch (scaling_strategy) {
	case SS_GLOBAL_AMAX:
	    return inplace_fake_quantize_block_scaling_gmax(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_BLOCK_AMAX:
	    return inplace_fake_quantize_block_scaling_bmax(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_E8:
	    return inplace_fake_quantize_block_scaling_e8(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
	case SS_CURRENT_SCALING:
	    return inplace_fake_quantize_current_scaling(input, num_rows, num_cols, block_num_rows, block_num_cols, e4m3_threshold);
	    break;
    }	
}

// Registers CUDA implementations for all kernels.
TORCH_LIBRARY_IMPL(mor, CUDA, m) {
  m.impl("fake_quantize", &fake_quantize_cuda);
  m.impl("fake_quantize_no_alloc", &fake_quantize_cuda_no_alloc);
  m.impl("quant_dequant_e4m3", &quant_dequant_e4m3);
  m.impl("quant_dequant_e5m2", &quant_dequant_e5m2);
  m.impl("quant_dequant_e3m2", &quant_dequant_e3m2);
  m.impl("quant_dequant_e2m3", &quant_dequant_e2m3);
  m.impl("e4m3_with_e8_scale_rne", &e4m3_with_e8_scale_rne);
  m.impl("e4m3_with_e8_scale_rz", &e4m3_with_e8_scale_rz);
  m.impl("fake_quantize_block_scaling", &fake_quantize_block_scaling);
  m.impl("fake_quantize_block_scaling_inplace", &inplace_fake_quantize_block_scaling);
}

}

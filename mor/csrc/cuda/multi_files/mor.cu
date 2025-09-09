// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

#include "mor.cuh"
#include "quantization_utils.cuh"
#include "warp_reduce.cuh"

template<ScalingFactorType T_SF_TYPE>
__device__ void get_block_error(const uint32_t* input_device, float* e4m3_sh, float* e5m2_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float block_amax, float block_amin) {
    float block_e4m3_error = 0;
    float block_e5m2_error = 0;
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

	    // Calculate E4M3 quantization error.
	    {
		float max_normal_fp8_e4m3 = 448;
		float scaling_factor_e4m3 = max_normal_fp8_e4m3/block_amax;
		float fp32_value1_quant_dequant_e4m3 = 0;
		float fp32_value2_quant_dequant_e4m3 = 0;
		if constexpr (T_SF_TYPE == SF_E8) {
		    fp32_value1_quant_dequant_e4m3 = SQDD_e8_e4m3_fp32(fp32_value1, scaling_factor_e4m3);
		    fp32_value2_quant_dequant_e4m3 = SQDD_e8_e4m3_fp32(fp32_value2, scaling_factor_e4m3);
		} else {
		    fp32_value1_quant_dequant_e4m3 = SQDD_fp32_e4m3_fp32(fp32_value1, scaling_factor_e4m3);
		    fp32_value2_quant_dequant_e4m3 = SQDD_fp32_e4m3_fp32(fp32_value2, scaling_factor_e4m3);
		}

		float error1_e4m3 = 0, error2_e4m3 = 0;
		if (fp32_value1) error1_e4m3 = fabsf((fp32_value1_quant_dequant_e4m3/fp32_value1) - 1);
		if (fp32_value2) error2_e4m3 = fabsf((fp32_value2_quant_dequant_e4m3/fp32_value2) - 1);
		block_e4m3_error += (error1_e4m3 + error2_e4m3);
	    }

	    // Calculate E5M2 quantization error.
	    {
		float max_normal_fp8_e5m2 = 57344;
		float scaling_factor_e5m2 = max_normal_fp8_e5m2/block_amax;
		float fp32_value1_quant_dequant_e5m2 = 0;
		float fp32_value2_quant_dequant_e5m2 = 0;
		if constexpr (T_SF_TYPE == SF_E8) {
		    fp32_value1_quant_dequant_e5m2 = SQDD_e8_e5m2_fp32(fp32_value1, scaling_factor_e5m2);
		    fp32_value2_quant_dequant_e5m2 = SQDD_e8_e5m2_fp32(fp32_value2, scaling_factor_e5m2);
		} else {
		    fp32_value1_quant_dequant_e5m2 = SQDD_fp32_e5m2_fp32(fp32_value1, scaling_factor_e5m2);
		    fp32_value2_quant_dequant_e5m2 = SQDD_fp32_e5m2_fp32(fp32_value2, scaling_factor_e5m2);
		}

		float error1_e5m2 = 0, error2_e5m2 = 0;
		if (fp32_value1) error1_e5m2 = fabsf((fp32_value1_quant_dequant_e5m2/fp32_value1) - 1);
		if (fp32_value2) error2_e5m2 = fabsf((fp32_value2_quant_dequant_e5m2/fp32_value2) - 1);
		block_e5m2_error += (error1_e5m2 + error2_e5m2);
	    }
	}
    }

    // Calculate the e4m3/e5m2 error for the thread block.
    // Step 1: Perform warp-level reduction within each warp
    block_e4m3_error = warpReduceSum(block_e4m3_error);
    block_e5m2_error = warpReduceSum(block_e5m2_error);

    // Step 2: For each warp, writes the per-warp amax/amin to shared memory
    // The 0th thread in a warp writes its amax/amin to shared memory.
    int flattened_tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((flattened_tid % warpSize) == 0) {
	e4m3_sh[flattened_tid / warpSize] = block_e4m3_error;
	e5m2_sh[flattened_tid / warpSize] = block_e5m2_error;
    }
    __syncthreads();

    // Step 3: For the first warp, read from the shared memory. Then reduce
    // in the warp.
    // We made an assumption here that the number of warps is <= warpSize.
    // Thus we only use one warp to read from the shared memory, get
    // warpSize elements from the shared memory, and then perform the
    // global reduction on that warp.
    if (flattened_tid < warpSize) {
	// Read the amax/amin from the shared memory.
	int num_warps = blockDim.x * blockDim.y / warpSize;
	block_e4m3_error = (flattened_tid < num_warps) ? e4m3_sh[flattened_tid] : 0;
	block_e5m2_error = (flattened_tid < num_warps) ? e5m2_sh[flattened_tid] : 0;

	// Perform the final warp-level reduction
	block_e4m3_error = warpReduceSum(block_e4m3_error);
	block_e5m2_error = warpReduceSum(block_e5m2_error);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
	e4m3_sh[0] = block_e4m3_error;
	e5m2_sh[0] = block_e5m2_error;
    }
    __syncthreads();
}

__device__ void get_block_amax_amin(const uint32_t* input_device, float* max_sh, float* min_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols) {
    // Compute max/min of the fp32 values.
    uint32_t bf16_max_binary = 0x7F7F;
    float bf16_max = bf16_to_fp32(bf16_max_binary);
    float block_amax = 0;
    float block_amin = bf16_max;

    // Partition one block (block_num_rows, block_num_cols) into many subblocks. Each sub-block has
    // size (tb_block_y, tb_block_x). Go over each sub-block.
    // X is the column ID. Y is the row ID.
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

	    // Initial Min/Max with clamp_threshold
	    float abs_value1 = fabsf(fp32_value1);
	    float abs_value2 = fabsf(fp32_value2);
	    float amax_value = max(abs_value1, abs_value2);
	    block_amax = max(block_amax, amax_value);
	    float amin_value = bf16_max;
	    if (abs_value1 > 0)
		amin_value = min(amin_value, abs_value1);
	    if (abs_value2 > 0)
		amin_value = min(amin_value, abs_value2);
	    if (amin_value > 0)
		block_amin = min(block_amin, amin_value);
	}
    }

    // Calculate the amax/amin for the thread block.
    // Step 1: Perform warp-level reduction within each warp
    block_amax = warpReduceMax(block_amax);
    block_amin = warpReduceMin(block_amin);

    // Step 2: For each warp, writes the per-warp amax/amin to shared memory
    // The purpose of the flattened_tid variable is to convert from the 2D
    // thread indices (x, y) to continuous 1D thread index.
    int flattened_tid = threadIdx.y * blockDim.x + threadIdx.x;
    // The 0th thread in a warp writes its amax/amin to shared memory.
    if ((flattened_tid % warpSize) == 0) {
        max_sh[flattened_tid / warpSize] = block_amax;
        min_sh[flattened_tid / warpSize] = block_amin;
    }
    __syncthreads();

    // Step 3: For the first warp, read from the shared memory. Then reduce
    // in the warp.
    // We made an assumption here that the number of warps is <= warpSize.
    // Thus we only use one warp to read from the shared memory, get
    // warpSize elements from the shared memory, and then perform the
    // global reduction on that warp.
    if (flattened_tid < warpSize) {
	// Read the amax/amin from the shared memory.
	int num_warps = blockDim.x * blockDim.y / warpSize;
	block_amax = (flattened_tid < num_warps) ? max_sh[flattened_tid] : -FLT_MAX;
	block_amin = (flattened_tid < num_warps) ? min_sh[flattened_tid] : FLT_MAX;

        // Perform the final warp-level reduction
        block_amax = warpReduceMax(block_amax);
        block_amin = warpReduceMin(block_amin);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
	max_sh[0] = block_amax;
	min_sh[0] = block_amin;
    }
    __syncthreads();
}

template <ScalingFactorType T_SF_TYPE>
__device__ MorQuantType choose_mor_quant_type(const uint32_t* input_device, float* e4m3_sh, float* e5m2_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float clamp_threshold, MorDecisionMode mor_mode, float block_amax, float block_amin) {
    float dynamic_range = 0;
    if (block_amax == 0)
    {
        block_amax = clamp_threshold;
        dynamic_range = 1;
    }

    if (dynamic_range == 0) dynamic_range = block_amax / block_amin;

    // The max/min normal number for E4M3 is 488/2^-6.
    // Reference https://arxiv.org/pdf/2209.05433
    // So we set the dynamic range for E4M3 in the normal range to be 448 * 2^6 = 28672.
    uint32_t e4m3_range = 28672;
    // We enlarge the E4M3 range by 4 to also assign 2 bits of subnormal range to E4M3
    // when we want to reduce the quantization error.
    if (mor_mode == QUANT_ERROR) {
        e4m3_range *= 4;
    }
    // The max/min normal number for E5M2 is 57344/2^-14.
    // Reference https://arxiv.org/pdf/2209.05433
    // So we set the dynamic range for E5M2 in the normal range to be 57344 * 2^14 = 939524096.
    uint32_t e5m2_range = 939524096;
    // The max/min normal number for E2M3 is 7.5/1.0.
    // Reference https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    // So we set the dynamic range for E2M3 in the normal range to be 7.5.
    float e2m3_range = 7.5;

    if (dynamic_range <= e2m3_range) {
        return QUANT_E2M3;
    }
    if (dynamic_range <= e4m3_range) {
        if (block_amax <= clamp_threshold)
            return QUANT_E4M3_SMALL;
        else
            return QUANT_E4M3;
    }
    else if (dynamic_range <= e5m2_range) {
        if (mor_mode == DYNAMIC_RANGE) {
            if (block_amax <= clamp_threshold)
                return QUANT_E5M2_SMALL;
            else
                return QUANT_E5M2;
        }
        // We need to sync here. This is because we need to ensure all the warps have
        // consumed the max/min in the shared memory, and we are able to reuse the same
        // shared memory for error calculations here.
        __syncthreads();
        get_block_error<T_SF_TYPE>(input_device, e4m3_sh, e5m2_sh, num_rows, num_cols, block_num_rows, block_num_cols, block_amax, block_amin);
        float block_e4m3_error = e4m3_sh[0];
        float block_e5m2_error = e5m2_sh[0];

        if (block_e4m3_error <= block_e5m2_error) {
            if (block_amax <= clamp_threshold)
                return QUANT_E4M3_SMALL;
            else
                return QUANT_E4M3;
        }
        else {
            // If E4M3 has higher error compared to E5M2, use bf16.
            return QUANT_BF16;
        }
    }
    return QUANT_BF16;
}

// Explicit template instantiations
template __device__ MorQuantType choose_mor_quant_type<SF_E8>(const uint32_t*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, float, MorDecisionMode, float, float);
template __device__ MorQuantType choose_mor_quant_type<SF_FP32>(const uint32_t*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, float, MorDecisionMode, float, float); 
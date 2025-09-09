// Standard C++ libraries.
#include <cassert>
#include <stdio.h>
#include <type_traits>
#include <vector>

// CUDA libraries.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// PyTorch libraries.
#include <torch/extension.h>

namespace cg = cooperative_groups;

// Macro to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                                         \
    {                                                                                  \
        cudaError_t err = (call);                                                      \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "           \
                      << __FILE__ << ":" << __LINE__ << std::endl;                     \
            exit(1);                                                                   \
        }                                                                              \
    }


__global__ void fake_quantize_block_scaling_kernel(
    const uint32_t* input_device,
    uint32_t* output_device,
    float* block_amax_device,
    int* block_non_zero_device,
    float* block_error_device,
    float* global_amax_device,
    int* global_non_zero_device,
    float* global_error_device,
    int* meta_device,
    int num_rows,
    int num_cols,
    int block_num_rows,
    int block_num_cols,
    int row_num_blocks,
    int col_num_blocks,
    float e4m3_threshold);

// Function to get the number of SMs on the device
int get_num_SMs() {
    int device;
    cudaGetDevice(&device);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    return numSMs;
}


__device__ float bf16_to_fp32(uint32_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    int32_t exponent = ((value >> 7) & 0xFF);
    uint32_t mantissa = value & 0x7F;
    uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 16);
    return *reinterpret_cast<float*>(&bits);
}

__device__ uint32_t fp32_to_bf16(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF);
    uint32_t mantissa = (bits >> 16) & 0x7F;
    uint32_t extra_mantissa = (bits >> 15) & 0x1;
    if (extra_mantissa == 1) {
        mantissa++;
	if (mantissa == 128) {
	    exponent++;
	    mantissa = 0;
	}
    }
    return (sign << 15) | (exponent << 7) | mantissa;
}

enum ScalingFactorType {
    SF_FP32 = 1,
    SF_E8 = 2
};

enum E8ScalingRoundingType {
    E8_RNE,
    E8_RZ
};

struct Exponent5 {
    static constexpr uint32_t num_bits = 5;
    static constexpr uint32_t bias = 15;
    static constexpr uint32_t before_overflow = 30;
    static constexpr uint32_t nan = 31;
    static constexpr uint32_t mask = 0x1F;
};

struct Exponent4 {
    static constexpr uint32_t num_bits = 4;
    static constexpr uint32_t bias = 7;
    static constexpr uint32_t before_overflow = 15;
    static constexpr uint32_t nan = 15;
    static constexpr uint32_t mask = 0xF;
};

struct Exponent3 {
    static constexpr uint32_t num_bits = 3;
    static constexpr uint32_t bias = 3;
    static constexpr uint32_t before_overflow = 7;
    // There is no NaN representation on E3M2
    // So when overflow happens, we stay with maximum value, 7.
    static constexpr uint32_t nan = 7;
    static constexpr uint32_t mask = 0x7;
};

struct Exponent2 {
    static constexpr uint32_t num_bits = 2;
    static constexpr uint32_t bias = 1;
    static constexpr uint32_t before_overflow = 3;
    // There is actuall no NaN representation on E2M3
    // So when overflow happens, we stay with maximum value, 3.
    static constexpr uint32_t nan = 3;
    static constexpr uint32_t mask = 0x3;
};

struct Mantissa3 {
    static constexpr uint32_t num_bits = 3;
    static constexpr uint32_t mask = 0x7;
    static constexpr uint32_t shift = 20;
    static constexpr uint32_t carry_over_value = 8;
    static constexpr uint32_t nan = 7;
};

struct Mantissa2 {
    static constexpr uint32_t num_bits = 2;
    static constexpr uint32_t mask = 0x3;
    static constexpr uint32_t shift = 21;
    static constexpr uint32_t carry_over_value = 4;
    static constexpr uint32_t nan = 3;
};


template <typename T_EXPONENT_TYPE,
	  typename T_MANTISSA_TYPE>
__device__ float quant_dequant_fp32(float fp32_value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&fp32_value);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + T_EXPONENT_TYPE::bias;
    uint32_t mantissa = 0;
    uint8_t guard_bit = 0;
    uint8_t sticky_bit = 0;

    if (exponent > 0) {
	// Normal range.
	mantissa = (bits >> T_MANTISSA_TYPE::shift) & T_MANTISSA_TYPE::mask;
	guard_bit = (bits >> (T_MANTISSA_TYPE::shift - 1)) & 0x1;
	uint32_t sticky_bit_mask = (1 << (T_MANTISSA_TYPE::shift - 1)) - 1;
	sticky_bit = ((bits & sticky_bit_mask) != 0);
    } else {
	// Subnormal range.
	uint32_t rshift = 1 - exponent;
	exponent = 0;
	if (rshift <= (T_MANTISSA_TYPE::num_bits + 1)) {
	    mantissa = bits & 0x7fffff | (1 << 23);
	    sticky_bit = ((mantissa & ((1 << (rshift + T_MANTISSA_TYPE::shift - 1)) - 1)) != 0);
	    mantissa >>= rshift;
	    guard_bit = (mantissa >> (T_MANTISSA_TYPE::shift - 1)) & 0x1;
	    mantissa = (mantissa >> (T_MANTISSA_TYPE::shift)) & T_MANTISSA_TYPE::mask;
	} else {
	    mantissa = 0;
	}
    }

    // Round to nearest even
    if (guard_bit == 1) {
	uint8_t mantissa_LSB = mantissa & 0x1;
	if (sticky_bit == 1 || mantissa_LSB == 1) {
	    mantissa++;
	    if (mantissa == T_MANTISSA_TYPE::carry_over_value) {
		mantissa = 0;
		exponent++;
	    }
	}
    }

    // For E3M2 and E2M3, if overflow happens, simply return the max value as there is no NaN.
    if constexpr (std::is_same<T_EXPONENT_TYPE, Exponent3>::value || std::is_same<T_EXPONENT_TYPE, Exponent2>::value) {
	if (exponent > T_EXPONENT_TYPE::before_overflow) {
	    exponent = (1 << T_EXPONENT_TYPE::num_bits) - 1;
	    mantissa = (1 << T_MANTISSA_TYPE::num_bits) - 1;
	}
    } else {
	// Overflow/Infinite/NaN cases for E5M2/E4M3.
	// Simply return NaN for all cases for simplicity.
	if (exponent > T_EXPONENT_TYPE::before_overflow) {
	    uint32_t kF32_NaN = 0x7fffffff;
	    return *reinterpret_cast<float*>(&kF32_NaN);
	} else if (exponent == T_EXPONENT_TYPE::before_overflow) {
	    if constexpr (std::is_same<T_EXPONENT_TYPE, Exponent4>::value) {
		// For E4M3, 1111 111 is NaN.
		if (mantissa == T_MANTISSA_TYPE::carry_over_value - 1) {
		    uint32_t kF32_NaN = 0x7fffffff;
		    return *reinterpret_cast<float*>(&kF32_NaN);
		}
	    }
	    // For E5M2, as the before_overflow is set to be 30, or 11110,
	    // any mantissa is allowed.
	}
    }

    // Convert back to fp32.
    uint32_t new_bits = 0;
    if (exponent > 0) {
	// Normal range.
	exponent += 127 - T_EXPONENT_TYPE::bias;
	new_bits = (sign << 31) | (exponent << 23) | (mantissa << T_MANTISSA_TYPE::shift);
    } else {
	// Subnormal range.
	if (mantissa) {
	    exponent += 127 - T_EXPONENT_TYPE::bias + 1;
	    uint32_t constexpr subnormal_mask = 1 << T_MANTISSA_TYPE::num_bits;
	    while ((mantissa & subnormal_mask) == 0) {
		mantissa <<= 1;
		exponent--;
	    }
	    mantissa &= T_MANTISSA_TYPE::mask;
	    new_bits = (sign << 31) | (exponent << 23) | (mantissa << T_MANTISSA_TYPE::shift);
	}
    }
    float pruned_fp32_value = *reinterpret_cast<float*>(&new_bits);

    return pruned_fp32_value;
}


template <int T_NUM_EXPONENTS, int T_NUM_MANTISSAS, bool T_USE_CUDA_CAST>
__device__ float quant_dequant_common(float value) {
    float output_value = 0;
    if constexpr (T_USE_CUDA_CAST) {
	if constexpr (T_NUM_EXPONENTS == 4 && T_NUM_MANTISSAS == 3) {
	    __nv_fp8_e4m3 quantized_value = static_cast<__nv_fp8_e4m3>(value);
	    output_value = static_cast<float>(quantized_value);
	} else if constexpr (T_NUM_EXPONENTS == 5 && T_NUM_MANTISSAS == 2) {
	    __nv_fp8_e5m2 quantized_value = static_cast<__nv_fp8_e5m2>(value);
	    output_value = static_cast<float>(quantized_value);
	} else {
	    assert(false);
	}
    } else {
	if constexpr (T_NUM_EXPONENTS == 4 && T_NUM_MANTISSAS == 3) {
	    output_value = quant_dequant_fp32<Exponent4, Mantissa3>(value);
	} else if constexpr (T_NUM_EXPONENTS == 5 && T_NUM_MANTISSAS == 2) {
	    output_value = quant_dequant_fp32<Exponent5, Mantissa2>(value);
	} else if constexpr (T_NUM_EXPONENTS == 3 && T_NUM_MANTISSAS == 2) {
	    output_value = quant_dequant_fp32<Exponent3, Mantissa2>(value);
	} else if constexpr (T_NUM_EXPONENTS == 2 && T_NUM_MANTISSAS == 3) {
	    output_value = quant_dequant_fp32<Exponent2, Mantissa3>(value);
	} else {
	    assert(false);
	}
    }
    return output_value;
}

// SQDD means scale, quantize, dequantize, and then descale.
template <E8ScalingRoundingType T_E8_SCALING_ROUNDING, int T_NUM_EXPONENTS, int T_NUM_MANTISSAS, bool T_USE_CUDA_CAST>
__device__ float SQDD_e8_scale(float value, float scaling_factor_fp32) {
    // Extract the exponent and mantissa for the scaling factor.
    uint32_t scaling_factor_bits = *reinterpret_cast<uint32_t*>(&scaling_factor_fp32);
    int32_t scaling_factor_e8 = ((scaling_factor_bits >> 23) & 0xFF) - 127;

    uint32_t value_bits = *reinterpret_cast<uint32_t*>(&value);
    uint32_t value_sign = (value_bits >> 31) & 0x1;
    int32_t value_exponent = ((value_bits >> 23) & 0xFF);
    uint32_t value_mantissa = value_bits & 0x7fffff;

    if constexpr (T_E8_SCALING_ROUNDING == E8_RNE) {
	assert(T_NUM_EXPONENTS == 4);
	assert(T_NUM_MANTISSAS == 3);
	uint32_t scaling_factor_mantissa = scaling_factor_bits & 0x7fffff;
    	uint8_t guard_bit = (scaling_factor_mantissa >> 22) & 0x1;
    	uint8_t sticky_bit = ((scaling_factor_mantissa & ((1 << 22) - 1)) != 0);
	// Calculate the e8 scaling factor.
	//
	// 6815744 is 11010000000000000000000 in binary.
	// E4M3 will take the first 3 bits from Mantissa (110)
	// The guard bit is 1 in this case, but the sticky bit is 0.
	// This is the tie case. For RNE, we will round to the nearest even.
	// The LSE of E4M3 is 0 in this case. So we will round down to 110.
	// Any mantissal larger than 6815744 will make the E4M3 to round up
	// to 111 and then becomes a NaN.
	if (guard_bit == 1 and value_mantissa <= 6815744 and value_exponent + scaling_factor_e8 < 135) {
	    uint8_t scaling_factor_e8_LSB = scaling_factor_e8 & 0x1;
	    // Round up the e8 scaling factor.
	    if (sticky_bit == 1 | scaling_factor_e8_LSB == 1) {
		scaling_factor_e8++;
	    }
	}
    }

    // Scale the value with e8 scaling factor.
    value_exponent += scaling_factor_e8;
    if (value_exponent < 0) value_exponent = 0;
    assert(value_exponent <= 254);
    uint32_t scaled_bits = (value_sign << 31) | (value_exponent << 23) | value_mantissa;
    float scaled_value = *reinterpret_cast<float*>(&scaled_bits);

    // Quantize and then dequantize.
    float dequant_value = quant_dequant_common<T_NUM_EXPONENTS, T_NUM_MANTISSAS, T_USE_CUDA_CAST>(scaled_value);

    // Descale.
    uint32_t dequant_value_bits = *reinterpret_cast<uint32_t*>(&dequant_value);
    uint32_t dequant_value_sign = (dequant_value_bits >> 31) & 0x1;
    int32_t dequant_value_exponent = ((dequant_value_bits >> 23) & 0xFF);
    uint32_t dequant_value_mantissa = dequant_value_bits & 0x7fffff;
    dequant_value_exponent -= scaling_factor_e8;
    // For negative small value, it can be quantized to -0.0, which is 2147483648 in uint32.
    // In such a case, dequant_value_exponent - scaling_factor_e8 can be < 0.
    if (dequant_value_exponent < 0) dequant_value_exponent = 0;
    assert(dequant_value_exponent <= 254);
    uint32_t descaled_bits = (dequant_value_sign << 31) | (dequant_value_exponent << 23) | dequant_value_mantissa;
    float descaled_value = *reinterpret_cast<float*>(&descaled_bits);
    return descaled_value;
}


// SQDD means scale, quantize, dequantize, and then descale.
template <int T_NUM_EXPONENTS, int T_NUM_MANTISSAS, bool T_USE_CUDA_CAST>
__device__ float SQDD_fp32_scale(float value, float scaling_factor) {
    float scaled_value = value * scaling_factor;
    float dequant_value = quant_dequant_common<T_NUM_EXPONENTS, T_NUM_MANTISSAS, T_USE_CUDA_CAST>(scaled_value);
    float output_value = dequant_value / scaling_factor;
    return output_value;
}

// SQDD means scale, quantize, dequantize, and then descale.
template <ScalingFactorType T_SCALING_TYPE, E8ScalingRoundingType T_E8_SCALING_ROUNDING, int T_NUM_EXPONENTS, int T_NUM_MANTISSAS, bool T_USE_CUDA_CAST, typename T_RETURN_TYPE>
__device__ T_RETURN_TYPE SQDD(float value, float scaling_factor) {
    float return_value_fp32 = 0.0;
    if constexpr (T_SCALING_TYPE == SF_E8) {
	return_value_fp32 = SQDD_e8_scale<T_E8_SCALING_ROUNDING, T_NUM_EXPONENTS, T_NUM_MANTISSAS, T_USE_CUDA_CAST>(value, scaling_factor);
    } else {
	return_value_fp32 = SQDD_fp32_scale<T_NUM_EXPONENTS, T_NUM_MANTISSAS, T_USE_CUDA_CAST>(value, scaling_factor);
    }

    if constexpr (std::is_same<T_RETURN_TYPE, uint32_t>::value) {
	uint32_t return_value_bf16 = fp32_to_bf16(return_value_fp32);
	return return_value_bf16;
    } else {
	return return_value_fp32;
    }
}

// Naming rule: SQDD_{ScalingFactorType}_{QuantizationType}_{ReturnType}
// Section 1: ScalingFactorType = SF_FP32, ReturnType = fp32
__device__ float SQDD_fp32_e4m3_fp32(float fp32_value, float scaling_factor) {
    return SQDD<SF_FP32, E8_RZ, 4, 3, true, float>(fp32_value, scaling_factor); 
}

__device__ float SQDD_fp32_e5m2_fp32(float fp32_value, float scaling_factor) {
    return SQDD<SF_FP32, E8_RZ, 5, 2, true, float>(fp32_value, scaling_factor); 
}

// Section 2: ScalingFactorType = SF_E8, ReturnType = fp32
__device__ float SQDD_e8_e4m3_fp32(float fp32_value, float scaling_factor) {
    return SQDD<SF_E8, E8_RZ, 4, 3, true, float>(fp32_value, scaling_factor); 
}

__device__ float SQDD_e8_e5m2_fp32(float fp32_value, float scaling_factor) {
    return SQDD<SF_E8, E8_RZ, 5, 2, true, float>(fp32_value, scaling_factor); 
}

// Section 3: ScalingFactorType = SF_FP32, ReturnType = bf16
__device__ uint32_t SQDD_fp32_e4m3_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_FP32, E8_RZ, 4, 3, true, uint32_t>(fp32_value, scaling_factor); 
}

__device__ uint32_t SQDD_fp32_e5m2_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_FP32, E8_RZ, 5, 2, true, uint32_t>(fp32_value, scaling_factor); 
}

__device__ uint32_t SQDD_fp32_e2m3_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_FP32, E8_RZ, 2, 3, false, uint32_t>(fp32_value, scaling_factor); 
}

// Section 4: ScalingFactorType = SF_E8, ReturnType = bf16
__device__ uint32_t SQDD_e8_e4m3_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_E8, E8_RZ, 4, 3, true, uint32_t>(fp32_value, scaling_factor); 
}

__device__ uint32_t SQDD_e8_e5m2_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_E8, E8_RZ, 5, 2, true, uint32_t>(fp32_value, scaling_factor); 
}

__device__ uint32_t SQDD_e8_e2m3_bf16(float fp32_value, float scaling_factor) {
    return SQDD<SF_E8, E8_RZ, 2, 3, false, uint32_t>(fp32_value, scaling_factor); 
}

// Helper function for warp-level reduction
__inline__ __device__ float warpReduceMax(float val)
{
    // Perform warp reduction for maximum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceMin(float val) {
    // Perform warp reduction for minimum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    // Perform warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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

template <ScalingFactorType T_SF_TYPE>
__device__ MorQuantType choose_mor_quant_type(const uint32_t* input_device, float* e4m3_sh, float* e5m2_sh, uint32_t num_rows, uint32_t num_cols, uint32_t block_num_rows, uint32_t block_num_cols, float clamp_threshold, MorDecisionMode mor_mode, float block_amax, float block_amin) {
    float dynamic_range = 0;
    if (block_amax == 0)
    {
	block_amax = clamp_threshold;
	dynamic_range = 1;
    }
    // else if (block_amax < 1e-25) block_amax = 1e-25;

    // if (block_amin < clamp_threshold) block_amin = clamp_threshold;

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
	//     if (block_amax <= clamp_threshold)
	// 	return QUANT_E5M2_SMALL;
	//     else
	// 	return QUANT_E5M2;
	}
    }
    return QUANT_BF16;
}

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

// Reduction helper function (amax and non-zero counts)
__device__ void blockReduceAmaxNonZero(float local_amax, int local_non_zero, float* s_amax, int* s_non_zero, int flattened_tid) {
    // Step 1: Perform warp-level reduction within each warp
    local_amax = warpReduceMax(local_amax);
    local_non_zero = warpReduceSum(local_non_zero);

    // Step 2: For each warp, writes the per-warp amax/amin to shared memory
    // The 0th thread in a warp writes its amax/amin to shared memory.
    if (flattened_tid % warpSize == 0) {
	s_amax[flattened_tid / warpSize] = local_amax;
	s_non_zero[flattened_tid / warpSize] = local_non_zero;
    }
    __syncthreads();

    // Step 3: For the first warp, read from the shared memory. Then reduce
    // in the warp.
    // We made an assumption here that the number of warps is <= warpSize.
    // Thus we only use one warp to read from the shared memory, get
    // warpSize elements from the shared memory, and then perform the
    // global reduction on that warp.
    if (flattened_tid < warpSize) {
	int num_warps = blockDim.x * blockDim.y / warpSize;
	local_amax = (flattened_tid < num_warps) ? s_amax[flattened_tid] : -FLT_MAX;
	local_non_zero = (flattened_tid < num_warps) ? s_non_zero[flattened_tid] : 0;

        // Perform the final warp-level reduction
	local_amax = warpReduceMax(local_amax);
	local_non_zero = warpReduceSum(local_non_zero);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
	s_amax[0] = local_amax;
	s_non_zero[0] = local_non_zero;
    }
    __syncthreads();
}

// Reduction helper function (relative error)
__device__ void blockReduceError(float local_error, float* s_error, int flattened_tid) {
    // Step 1: Perform warp-level reduction within each warp
    local_error = warpReduceSum(local_error);

    // Step 2: For each warp, writes the per-warp amax/amin to shared memory
    // The 0th thread in a warp writes its amax/amin to shared memory.
    if (flattened_tid % warpSize == 0) {
	s_error[flattened_tid / warpSize] = local_error;
    }
    __syncthreads();

    // Step 3: For the first warp, read from the shared memory. Then reduce
    // in the warp.
    // We made an assumption here that the number of warps is <= warpSize.
    // Thus we only use one warp to read from the shared memory, get
    // warpSize elements from the shared memory, and then perform the
    // global reduction on that warp.
    if (flattened_tid < warpSize) {
	int num_warps = blockDim.x * blockDim.y / warpSize;
	local_error = (flattened_tid < num_warps) ? s_error[flattened_tid] : 0;

        // Perform the final warp-level reduction
	local_error = warpReduceSum(local_error);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
	s_error[0] = local_error;
    }
    __syncthreads();
}

// Combined kernel for Phases 1 and 2
__global__ void fake_quantize_block_scaling_kernel(
    const uint32_t* input_device,
    uint32_t* output_device,
    float* block_amax_device,
    int* block_non_zero_device,
    float* block_error_device,
    float* global_amax_device,
    int* global_non_zero_device,
    float* global_error_device,
    int* meta_device,
    int num_rows,
    int num_cols,
    int block_num_rows,
    int block_num_cols,
    int row_num_blocks,
    int col_num_blocks,
    float e4m3_threshold) {
    int threads_per_block = blockDim.x * blockDim.y;
    int warps_per_block = threads_per_block / warpSize;

    int total_sub_tensors = col_num_blocks * row_num_blocks;
    extern __shared__ float sharedMem[];

    float* s_amax = sharedMem;
    int* s_non_zero = (int*)((char*)sharedMem + (warps_per_block * sizeof(float)));

    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Compute start and end indices for sub-tensors assigned to this block
    int num_blocks = gridDim.x * gridDim.y;
    int sub_tensors_per_block = (total_sub_tensors + num_blocks - 1) / num_blocks;

    int start_sub_tensor_id = (block_id + 0) * sub_tensors_per_block;
    int   end_sub_tensor_id = (block_id + 1) * sub_tensors_per_block;

    if (start_sub_tensor_id > total_sub_tensors) start_sub_tensor_id = total_sub_tensors;
    if (  end_sub_tensor_id > total_sub_tensors)   end_sub_tensor_id = total_sub_tensors;

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 1: Compute local block maxima
    //////////////////////////////////////////////////////////////////////////////////
    for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id) 
    {
        int sub_tensor_row = sub_tensor_id / col_num_blocks;
        int sub_tensor_col = sub_tensor_id % col_num_blocks;

        int globalBlockRow = sub_tensor_row * block_num_rows;
        int globalBlockCol = sub_tensor_col * block_num_cols;

        float local_amax = -INFINITY;
        int local_non_zero = 0;

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
            block_amax_device[sub_tensor_id] = s_amax[0];
	    block_non_zero_device[sub_tensor_id] = s_non_zero[0];
        }
    }

    grid.sync();  // Ensure all blocks have written their block-level maxima and non-zero counts

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 2: Compute global maxima and non-zero counts
    //////////////////////////////////////////////////////////////////////////////////

    float global_amax = -FLT_MAX;
    int global_non_zero = 0;

    for (int i = tid; i < total_sub_tensors; i += blockDim.x * blockDim.y) {
        global_amax = fmaxf(global_amax, block_amax_device[i]);
	global_non_zero += block_non_zero_device[i];
    }
    
    blockReduceAmaxNonZero(global_amax, global_non_zero, s_amax, s_non_zero, tid);

    global_amax = s_amax[0];
    global_non_zero = s_non_zero[0];
    
    if (block_id == 0 && tid == 0) {
	global_amax_device[0] = global_amax;
	global_non_zero_device[0] = global_non_zero;
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 3: Quantize the subtensors to E4M3 and compute subtensor relative error
    //////////////////////////////////////////////////////////////////////////////////

    uint32_t matrix_sf_mantissa = 0;
    {
        float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
        float ratio = max_fp8_e4m3 / global_amax;
        matrix_sf_mantissa = *reinterpret_cast<uint32_t*>(&ratio) & 0x007FFFFF;
    }


    // Iterate through the assigned sub-tensors for this block
    for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id)
    {
        int sub_tensor_row = sub_tensor_id / col_num_blocks;
        int sub_tensor_col = sub_tensor_id % col_num_blocks;

        int globalBlockRow = sub_tensor_row * block_num_rows;
        int globalBlockCol = sub_tensor_col * block_num_cols;

        // Compute scaling factor for this sub-tensor
        float scaling_factor = 1.0; //initialize it to something...

	{
	    float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
	    float block_amax = block_amax_device[sub_tensor_id];
	    float block_sf = max_fp8_e4m3 / block_amax;
	    uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
	    uint32_t block_sf_mantissa = block_sf_bits & 0x007FFFFF;
	    uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
	    float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
	    scaling_factor = (matrix_sf_mantissa <= block_sf_mantissa) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;
	    assert(scaling_factor < 1e30);
	}

        // Quantize values in the sub-tensor
        float local_error = 0.0;

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
                    dequantized_value1 = SQDD_fp32_e4m3_fp32(fp32_value1, scaling_factor);
                    float value_error = fabsf((dequantized_value1 - fp32_value1) / fp32_value1);
                    local_error += value_error;
                }

                if (fp32_value2 != 0)
                {
                    dequantized_value2 = SQDD_fp32_e4m3_fp32(fp32_value2, scaling_factor);
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
        blockReduceError(local_error, s_amax, tid);

        // Write results to global memory
        if (tid == 0) 
        {
            block_error_device[sub_tensor_id] = s_amax[0];
        }
    }

    grid.sync();  // Ensure all blocks have written their block-level maxima and non-zero counts

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 4: Compute global relative error
    //////////////////////////////////////////////////////////////////////////////////
    float global_error = 0;

    for (int i = tid; i < total_sub_tensors; i += blockDim.x * blockDim.y) 
    {
        global_error += block_error_device[i];
    }

    blockReduceError(global_error, s_amax, tid);
    global_error = s_amax[0];

    if (block_id == 0 && tid == 0) {
	global_error_device[0] = global_error;
    }

    float average_relative_error = global_error/global_non_zero;

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 5: Fall back to BF16 if the relative error is too high
    //////////////////////////////////////////////////////////////////////////////////
    if (average_relative_error > e4m3_threshold) {
	if (block_id == 0 && tid == 0) {
	    meta_device[0] = 3;
	}

        //We are going to store in BF16, and basically copy from A to B :)
        for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id) 
        {
            int sub_tensor_row = sub_tensor_id / col_num_blocks;
            int sub_tensor_col = sub_tensor_id % col_num_blocks;

            int globalBlockRow = sub_tensor_row * block_num_rows;
            int globalBlockCol = sub_tensor_col * block_num_cols;

            for (int r = 0; r < block_num_rows; r += blockDim.y)
            {
                int row = globalBlockRow + r + threadIdx.y;

                for (int c = 0; c < block_num_cols; c += blockDim.x)
                {
                    int col = globalBlockCol + c + threadIdx.x;
                    output_device[row * num_cols + col] = input_device[row * num_cols + col];
                }
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

int getAdaptiveBlocksForCooperativeKernel(void* kernel, int blockSize) {
    int numSMs, maxBlocksPerSM;

    // Get total number of SMs
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    numSMs = prop.multiProcessorCount;

    // Get max active blocks per SM for the given kernel
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, (void*)kernel, blockSize, 0));

    // Assume 15% of GPU resources are used → Only 85% available
    float availableFraction = 0.75f;

    // Compute the total blocks we can launch (adjusted for assumed GPU usage)
    int maxBlocksTotal = static_cast<int>(maxBlocksPerSM * numSMs * availableFraction);

    return maxBlocksTotal > 0 ? maxBlocksTotal : 1;
}

std::vector<at::Tensor> fake_quantize_block_scaling(const at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
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
    int total_sub_tensors = col_num_blocks * row_num_blocks;

    float *block_amax;
    int *block_non_zero;
    float *block_error;

    CHECK_CUDA_ERROR(cudaMalloc(&block_amax, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&block_non_zero,  total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&block_error, total_sub_tensors * sizeof(float)));

    int threads_x = 32;
    int threads_y = 8;

    // int numSMs = get_num_SMs();
    // int numBlocks = numSMs;  // Default: 1 thread block per SM
    // Ensure that the number of thread blocks is a power of 2.
    // Each SM will have up to 2 thread blocks.
    // if (total_sub_tensors % numBlocks) numBlocks = pow(2, floor(log2(numBlocks)) + 1);
    int blockSize = threads_x * threads_y;
    int numBlocks = getAdaptiveBlocksForCooperativeKernel((void*)fake_quantize_block_scaling_kernel, blockSize);


    // printf("row_packed: %d, col_packed: %d, num_SMs: %d, numBlocks: %d, total_sub_tensors: %d\n", row_packed, col_packed, numSMs, numBlocks, total_sub_tensors);
    // fflush(stdout);

    int threads_per_block = threads_x * threads_y;
    TORCH_CHECK((threads_per_block % 32) == 0);
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim(min(numBlocks, total_sub_tensors), 1);  // Ensure grid fits total sub-tensors

    int warps_per_block = threads_per_block / 32;

    int sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    float e4m3_threshold_fp32 = static_cast<float>(e4m3_threshold);
    void* kernelArgs[] = {
        (void*)&input_uint32_ptr,
	(void*)&result_uint32_ptr,
	(void*)&block_amax,
	(void*)&block_non_zero,
	(void*)&block_error,
	(void*)&global_amax_ptr,
	(void*)&global_non_zero_ptr,
	(void*)&global_error_ptr,
	(void*)&meta_ptr,
        (void*)&row_packed,
	(void*)&col_packed, 
        (void*)&block_num_rows_packed,
	(void*)&block_num_cols_packed, 
	(void*)&row_num_blocks,
        (void*)&col_num_blocks,
	(void*)&e4m3_threshold_fp32
    };

    // Launch the kernel using cooperative kernel launch
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel( (void*)fake_quantize_block_scaling_kernel, gridDim, blockDim, kernelArgs, sharedMemSize));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaFree(block_amax));
    CHECK_CUDA_ERROR(cudaFree(block_non_zero));
    CHECK_CUDA_ERROR(cudaFree(block_error));

    return output;
}


std::vector<at::Tensor> inplace_fake_quantize_block_scaling(at::Tensor& input, int64_t num_rows, int64_t num_cols, int64_t block_num_rows, int64_t block_num_cols, double e4m3_threshold) {
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
    int total_sub_tensors = col_num_blocks * row_num_blocks;

    uint32_t *result_uint32_ptr;
    float *block_amax;
    int *block_non_zero;
    float *block_error;

    CHECK_CUDA_ERROR(cudaMalloc(&result_uint32_ptr, row_packed * col_packed * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&block_amax, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&block_non_zero,  total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&block_error, total_sub_tensors * sizeof(float)));

    int threads_x = 32;
    int threads_y = 8;

    // int numSMs = get_num_SMs();
    // int numBlocks = numSMs;  // Default: 1 thread block per SM
    // Ensure that the number of thread blocks is a power of 2.
    // Each SM will have up to 2 thread blocks.
    // if (total_sub_tensors % numBlocks) numBlocks = pow(2, floor(log2(numBlocks)) + 1);
    int blockSize = threads_x * threads_y;
    int numBlocks = getAdaptiveBlocksForCooperativeKernel((void*)fake_quantize_block_scaling_kernel, blockSize);


    // printf("row_packed: %d, col_packed: %d, num_SMs: %d, numBlocks: %d, total_sub_tensors: %d\n", row_packed, col_packed, numSMs, numBlocks, total_sub_tensors);
    // fflush(stdout);

    int threads_per_block = threads_x * threads_y;
    TORCH_CHECK((threads_per_block % 32) == 0);
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim(min(numBlocks, total_sub_tensors), 1);  // Ensure grid fits total sub-tensors

    int warps_per_block = threads_per_block / 32;

    int sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    float e4m3_threshold_fp32 = static_cast<float>(e4m3_threshold);
    void* kernelArgs[] = {
        (void*)&input_uint32_ptr,
	(void*)&result_uint32_ptr,
	(void*)&block_amax,
	(void*)&block_non_zero,
	(void*)&block_error,
	(void*)&global_amax_ptr,
	(void*)&global_non_zero_ptr,
	(void*)&global_error_ptr,
	(void*)&meta_ptr,
        (void*)&row_packed,
	(void*)&col_packed,
        (void*)&block_num_rows_packed,
	(void*)&block_num_cols_packed,
	(void*)&row_num_blocks,
        (void*)&col_num_blocks,
	(void*)&e4m3_threshold_fp32
    };

    // Launch the kernel using cooperative kernel launch
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel( (void*)fake_quantize_block_scaling_kernel, gridDim, blockDim, kernelArgs, sharedMemSize));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(input_uint32_ptr, result_uint32_ptr, row_packed * col_packed * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaFree(block_amax));
    CHECK_CUDA_ERROR(cudaFree(block_non_zero));
    CHECK_CUDA_ERROR(cudaFree(block_error));
    CHECK_CUDA_ERROR(cudaFree(result_uint32_ptr));

    return output;
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

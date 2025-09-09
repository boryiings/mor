#pragma once

// Standard C++ libraries.
#include <cassert>
#include <type_traits>

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

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
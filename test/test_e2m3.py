import torch
import unittest
import struct
import random

# ==============================================================================
#  The Generic Casting Function from the Previous Step
# ==============================================================================

def cast_bf16_subnormal_to_custom_fp(bf16_int, total_bits, exponent_bits, mantissa_bits):
    """
    Casts a bf16 number in a target format's subnormal range to that custom
    floating-point format using round to nearest, ties to even rounding.
    """
    # 1. Extract components from bf16 input
    sign_bf16 = (bf16_int >> 15) & 0x1
    exp_bf16 = (bf16_int >> 7) & 0xFF
    mant_bf16 = bf16_int & 0x7F
    bf16_exp_bias = 127

    # 2. Define target format parameters
    target_exp_bias = (1 << (exponent_bits - 1)) - 1
    min_normal_exp = 1 - target_exp_bias

    # 3. Calculate alignment shift
    mant_bf16_with_implicit_one = mant_bf16 | 0x80
    shift = min_normal_exp - (exp_bf16 - bf16_exp_bias)

    if shift < 0:
        smallest_normal_exp = 1
        smallest_normal_mant = 0
        sign_shift = total_bits - 1
        return (sign_bf16 << sign_shift) | (smallest_normal_exp << mantissa_bits) | smallest_normal_mant

    aligned_mant = mant_bf16_with_implicit_one >> shift

    # 4. Perform Round to Nearest, Ties to Even
    if shift > 0:
        guard_pos = shift - 1
        guard_bit = (mant_bf16_with_implicit_one >> guard_pos) & 1

        round_pos = shift - 2
        round_bit = (mant_bf16_with_implicit_one >> round_pos) & 1 if round_pos >= 0 else 0
        
        sticky_mask = (1 << round_pos) - 1 if round_pos > 0 else 0
        sticky_bit = 1 if (mant_bf16_with_implicit_one & sticky_mask) != 0 else 0

        if guard_bit == 1 and (round_bit == 1 or sticky_bit == 1):
            aligned_mant += 1
        elif guard_bit == 1 and round_bit == 0 and sticky_bit == 0: # Tie case
            if (aligned_mant & 1) == 1:
                aligned_mant += 1

    # 5. Construct the final number
    target_mant_mask = (1 << mantissa_bits) - 1
    target_mant = aligned_mant & target_mant_mask
    
    if (aligned_mant >> mantissa_bits) & 1:
        target_exp = 1
        target_mant = 0
    else:
        target_exp = 0

    sign_shift = total_bits - 1
    return (sign_bf16 << sign_shift) | (target_exp << mantissa_bits) | target_mant

def cast_bf16_subnormal_to_fp8_e4m3_generic(bf16_int):
    """Wrapper for E4M3 conversion."""
    return cast_bf16_subnormal_to_custom_fp(
        bf16_int=bf16_int, total_bits=8, exponent_bits=4, mantissa_bits=3
    )

def cast_bf16_subnormal_to_fp8_e2m3_generic(bf16_int):
    """Wrapper for E4M3 conversion."""
    return cast_bf16_subnormal_to_custom_fp(
        bf16_int=bf16_int, total_bits=6, exponent_bits=2, mantissa_bits=3
    )

# ==============================================================================
#  Unit Test Class
# ==============================================================================

class TestE4M3SubnormalConversion(unittest.TestCase):

    def float_to_bf16_int(self, f):
        """Converts a Python float to a bf16 integer representation."""
        # PyTorch's bfloat16 is not a true 16-bit type in Python,
        # so we convert to float32, get bytes, and take the top 16 bits.
        b = struct.pack('!f', f)
        return struct.unpack('!H', b[:2])[0]

    def test_subnormal_range_casting(self):
        """
        Tests casting of bf16 numbers in the E4M3 subnormal range
        against PyTorch's implementation.
        """
        # E4M3 parameters
        E4M3_EXPONENT_BIAS = 7
        E4M3_MANTISSA_BITS = 3
        
        # The smallest positive normal number in E4M3 is 2**(1 - bias) * (1.0)
        # = 2**(-6) approx 0.015625
        min_normal_e4m3 = 2**(1 - E4M3_EXPONENT_BIAS)

        # The largest subnormal number in E4M3 is 2**(-6) * (0.111)_2
        # = 2**(-6) * (7/8) = 0.013671875
        max_subnormal_e4m3 = min_normal_e4m3 * (1 - 2**-E4M3_MANTISSA_BITS)
        
        # We will sample bf16 numbers between 0 and this max value.
        
        num_samples = 5000  # Test a good number of random values
        
        for i in range(num_samples):
            # Generate a random float in the E4M3 subnormal range.
            # We also test negative numbers.
            sign = 1 if random.random() < 0.5 else -1
            random_float = random.uniform(0, max_subnormal_e4m3) * sign

            with self.subTest(f"Testing float value: {random_float}"):
                # 1. Get the bf16 integer representation of our test float.
                bf16_input_int = self.float_to_bf16_int(random_float)

                # 2. Use our custom implementation to cast to E4M3.
                custom_result_int = cast_bf16_subnormal_to_fp8_e4m3_generic(bf16_input_int)

                # 3. Use PyTorch to perform the same casting.
                #    - Create a float tensor
                #    - Cast it to bfloat16
                #    - Cast it to float8_e4m3fn
                float_tensor = torch.tensor([random_float], dtype=torch.float32)
                bf16_tensor = float_tensor.to(torch.bfloat16)
                e4m3_tensor = bf16_tensor.to(torch.float8_e4m3fn)
                
                # Extract the underlying integer value from the PyTorch tensor.
                # The `.item()` gets the Python value, which for float8 is just a float.
                # We need to access its raw byte representation.
                pytorch_result_bytes = e4m3_tensor.char().numpy().tobytes()
                pytorch_result_int = int.from_bytes(pytorch_result_bytes, 'little')

                # 4. Compare the results.
                self.assertEqual(custom_result_int, pytorch_result_int,
                                 f"Failed for float: {random_float}, "
                                 f"bf16 int: {bf16_input_int:#06x}. "
                                 f"Custom: {custom_result_int:#04x}, "
                                 f"PyTorch: {pytorch_result_int:#04x}")

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()

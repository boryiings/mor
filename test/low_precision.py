def bf16_to_fp8_subnormal_e4m3(bf16_val):
    """
    Casts a bf16 floating-point number to an E4M3 (FP8) number,
    assuming the bf16 value is in the subnormal range and the result
    will also be in the subnormal range of E4M3.

    E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    bf16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits.

    This function focuses specifically on the subnormal handling using
    binary operations. Subnormal numbers have a zero exponent and an
    implied leading zero before the mantissa.

    Args:
        bf16_val (int): The bf16 number as an integer (16-bit representation).

    Returns:
        int: The E4M3 number as an integer (8-bit representation),
             or None if the bf16 input is not a subnormal number
             that would result in a subnormal E4M3.
    """
    if not isinstance(bf16_val, int) or not (0 <= bf16_val <= 0xFFFF):
        raise ValueError("bf16_val must be a 16-bit unsigned integer.")

    # Extract sign bit
    sign_bf16 = (bf16_val >> 15) & 0x1

    # Extract exponent and mantissa for bf16
    exponent_bf16 = (bf16_val >> 7) & 0xFF  # 8 bits
    mantissa_bf16 = bf16_val & 0x7F         # 7 bits

    # Check if bf16 is a subnormal number (exponent is 0, mantissa is not 0)
    if exponent_bf16 != 0:
        # If it's a normal bf16, this function's scope is not met.
        # We would need to handle normal range casting separately.
        # For simplicity, we return None or raise an error.
        return None  # Or raise ValueError("Input bf16 is not a subnormal number.")

    if exponent_bf16 == 0 and mantissa_bf16 != 0:  # This is a bf16 subnormal
        # For standard value-preserving casting, it underflows to 0 for E4M3.
        # Check sign and return 0.
        return sign_bf16 << 7  # 0 for positive, 0x80 for negative. (Sign bit in E4M3)
                               # Or simply 0 if we return unsigned int.

    # If the bf16 is 0, it maps to 0.
    if bf16_val == 0 or bf16_val == 0x8000: # Positive or negative zero
        return sign_bf16 << 7 # E4M3 zero has sign bit only (0 or 0x80)

    # Check if bf16 is a subnormal (exponent is 0, mantissa is not 0)
    if exponent_bf16 == 0:
        # If mantissa is 0, it's zero, which maps to E4M3 zero.
        if mantissa_bf16 == 0:
            return sign_bf16 << 7

        # This part implements a conceptual mapping of bf16 subnormal mantissa
        # to E4M3 subnormal mantissa, ignoring the massive exponent difference
        # that would cause underflow in a standard F-P conversion.
        # This is for demonstration of bit manipulation for *mantissa-level* subnormal handling.

        # The bf16 mantissa is 7 bits (0 to 127).
        # The E4M3 mantissa is 3 bits (0 to 7).
        # We need to effectively right-shift the bf16 mantissa by 4 bits (7 - 3 = 4).

        # For rounding (to nearest, ties to even):
        # We need to examine the 4 least significant bits of the bf16 mantissa (`mantissa_bf16 & 0xF`).
        # The bit at position 3 (0-indexed) is the first bit we "discard".
        # If the discarded part is > 0.5 (i.e., `mantissa_bf16 & 0xF` is > 8), we round up.
        # If the discarded part is exactly 0.5 (i.e., `mantissa_bf16 & 0xF` is 8), we round to nearest even.

        # Get the rounded mantissa for E4M3
        e4m3_mantissa = mantissa_bf16 >> 4  # Truncate to 3 bits (most significant)

        # Check for rounding condition (bits m3, m2, m1, m0)
        # The value of these 4 bits determines rounding.
        # We are looking at `mantissa_bf16[3:0]`
        discarded_fraction = mantissa_bf16 & 0xF # The 4 lowest bits
        
        # Round to nearest, ties to even:
        # If discarded_fraction > 0x8 (more than half), round up.
        # If discarded_fraction == 0x8 (exactly half), round up if the current e4m3_mantissa is odd.
        if discarded_fraction > 0x8: # 0x8 = 0b1000
            e4m3_mantissa += 1
        elif discarded_fraction == 0x8:
            # Tie case: round to even. If `e4m3_mantissa` is odd, increment it.
            if e4m3_mantissa & 0x1: # Check if the least significant bit is 1 (odd)
                e4m3_mantissa += 1

        # Handle overflow in mantissa (e.g., if rounding causes it to exceed 7).
        # In E4M3 subnormals, mantissa is 0 to 7. If it becomes 8 (0b1000), it's effectively 0,
        # but this would mean it transitioned to a normal number, which is out of scope for
        # this *subnormal-to-subnormal* bit manipulation scenario.
        # For actual FP conversion, this would be a transition to normal range.
        # Here, we constrain it to 3 bits.
        e4m3_mantissa &= 0x7 # Ensure it stays within 3 bits (0-7)

        # The exponent for E4M3 subnormals is always 0.
        # The sign bit is `sign_bf16`.
        e4m3_val = (sign_bf16 << 7) | (0x0 << 3) | e4m3_mantissa
        return e4m3_val
    else:
        # If the bf16 is a normal number, NaN, or Inf, this function doesn't apply
        return None # Or raise an error


def bf16_to_fp6_subnormal_e2m3(bf16_val):
    """
    Casts a bf16 floating-point number to an E2M3 (FP6) number,
    specifically handling the case where the bf16 value is in the
    subnormal range.

    E2M3 format: 1 sign bit, 2 exponent bits, 3 mantissa bits.
    bf16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits.

    Args:
        bf16_val (int): The bf16 number as an integer (16-bit representation).

    Returns:
        int: The E2M3 number as an integer (6-bit representation),
             or None if the bf16 input is not a subnormal number,
             or if it's a subnormal that underflows to zero (standard conversion).
    """
    if not isinstance(bf16_val, int) or not (0 <= bf16_val <= 0xFFFF):
        raise ValueError("bf16_val must be a 16-bit unsigned integer.")

    # Extract sign bit
    sign_bf16 = (bf16_val >> 15) & 0x1

    # Extract exponent and mantissa for bf16
    exponent_bf16 = (bf16_val >> 7) & 0xFF  # 8 bits
    mantissa_bf16 = bf16_val & 0x7F         # 7 bits

    # --- Standard Floating-Point Conversion Handling ---
    # If bf16 is zero, map to E2M3 zero.
    if bf16_val == 0 or bf16_val == 0x8000: # Positive or negative zero
        return sign_bf16 << 5 # E2M3 zero has sign bit only (0 or 0x20)

    # Check if bf16 is a subnormal number (exponent is 0, mantissa is not 0)
    if exponent_bf16 == 0 and mantissa_bf16 != 0:
        # As discussed, bf16 subnormals are orders of magnitude smaller than E2M3 subnormals.
        # They will almost certainly underflow to zero in a standard value-preserving conversion.
        # If the intent is strict floating-point conversion, uncomment the line below:
        # return sign_bf16 << 5 # Return 0 for positive, 0x20 for negative.

        # --- Optional: Hypothetical Mantissa Bit-Mapping with Rounding ---
        # This section demonstrates the bit manipulation for rounding *if*
        # one were to conceptually "map" bf16 subnormal mantissas to E2M3
        # subnormal mantissas, *ignoring* the massive exponent difference that
        # causes underflow in real FP conversion. This is a non-standard interpretation
        # for a direct value conversion but illustrates the requested "binary operations"
        # for "subnormal range handling" at the mantissa level.

        # bf16 mantissa is 7 bits. E2M3 mantissa is 3 bits.
        # We need to effectively right-shift by 7 - 3 = 4 bits to get the most significant 3 bits.

        # Calculate the initial truncated mantissa
        e2m3_mantissa = mantissa_bf16 >> 4

        # Check for rounding (to nearest, ties to even)
        # We need to examine the 4 least significant bits of the bf16 mantissa (`mantissa_bf16 & 0xF`).
        # These are the bits m3, m2, m1, m0 (0-indexed from the right of the 7-bit mantissa).
        # The first bit being discarded is m3.
        
        discarded_fraction = mantissa_bf16 & 0xF # The 4 lowest bits (0b_m3_m2_m1_m0)

        # Round to nearest, ties to even:
        # Rule 1: If the discarded bits represent a value strictly greater than 0.5 (i.e., > 0b1000 or 8), round up.
        # Rule 2: If the discarded bits represent a value exactly 0.5 (i.e., 0b1000 or 8), round to nearest even.
        #         This means if the current e2m3_mantissa (the retained part) is odd, we round up to make it even.
        #         If it's already even, we don't round up.

        if discarded_fraction > 0x8: # If the fractional part is > 0.5 (e.g., 0.1001, 0.1111)
            e2m3_mantissa += 1
        elif discarded_fraction == 0x8: # If the fractional part is exactly 0.5 (e.g., 0.1000)
            # Check if the current (truncated) e2m3_mantissa is odd.
            # If it's odd (least significant bit is 1), increment to make it even.
            if e2m3_mantissa & 0x1:
                e2m3_mantissa += 1

        # Constrain mantissa to 3 bits (0-7).
        # If rounding caused it to become 8 (0b1000), it means it would "overflow" the subnormal mantissa range
        # and conceptually become the smallest normal number. For this *subnormal-to-subnormal* mapping,
        # we'll clamp it to the max subnormal mantissa.
        e2m3_mantissa &= 0x7 # Ensure it stays within 3 bits (0-7)

        # E2M3 subnormal has exponent bits 0b00.
        fp6_val = (sign_bf16 << 5) | (0x0 << 3) | e2m3_mantissa
        return fp6_val
    # --- End Optional Section ---

    # If it's not a bf16 subnormal (e.g., normal or NaN/Inf), this function doesn't apply.
    return None # Or raise an error


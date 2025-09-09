# Python imports
import math
import struct
import unittest

# Library imports
import numpy as np
import torch

# Project imports
import mor

def float_to_int_value(float_value):
    # Pack the float into a byte string
    packed_float = struct.pack("<f", float_value)
    # Unpack the byte string as an integer
    int_value = struct.unpack("<i", packed_float)[0]
    return int_value

def int_to_float(int_value):
    # Pack the integer into a byte string
    packed_int = struct.pack("<i", int_value) # '<i' for little-endian integer
    # Unpack the byte string as a float
    float_value = struct.unpack("<f", packed_int)[0] # '<f' for little-endian float
    return float_value

def decompose_float(value):
    bin_value = float_to_int_value(value)
    sign = bin_value >> 31
    exponent = (bin_value >> 23) & 0xFF
    mantissa = (bin_value >> 16) & 0x7F
    return sign, exponent, mantissa

def e8m1_quant(value):
    negative = False
    if value < 0:
        negative = True
        abs_value = value * -1
    else:
        abs_value = value
    sign, exponent, mantissa = decompose_float(abs_value)
    if exponent == 0:
        mantissa = 0
    elif exponent < 255:
        if mantissa <= 32:
            mantissa = 0
        elif mantissa <= 95:
            mantissa = 1 << 6
        else:
            mantissa = 0
            exponent += 1
            if exponent == 255:
                exponent = 254
                mantissa = 1 << 6
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16)
    float_value = int_to_float(bits)
    if negative:
        float_value = float_value * -1
    return float_value

def e8m2_quant(value):
    negative = False
    if value < 0:
        negative = True
        abs_value = value * -1
    else:
        abs_value = value
    sign, exponent, mantissa = decompose_float(abs_value)
    if exponent == 0:
        mantissa = 0
    elif exponent < 255:
        if mantissa <= 16:
            mantissa = 0
        elif mantissa <= 47:
            mantissa = 1 << 5
        elif mantissa <= 80:
            mantissa = 2 << 5
        elif mantissa <= 111:
            mantissa = 3 << 5
        else:
            mantissa = 0
            exponent += 1
            if exponent == 255:
                exponent = 254
                mantissa = 3 << 5
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16)
    float_value = int_to_float(bits)
    if negative:
        float_value = float_value * -1
    return float_value

def e8m3_quant(value):
    negative = False
    if value < 0:
        negative = True
        abs_value = value * -1
    else:
        abs_value = value
    sign, exponent, mantissa = decompose_float(abs_value)
    if exponent == 0:
        mantissa = 0
    elif exponent < 255:
        if mantissa <= 8:
            mantissa = 0
        elif mantissa <= 23:
            mantissa = 1 << 4
        elif mantissa <= 40:
            mantissa = 2 << 4
        elif mantissa <= 55:
            mantissa = 3 << 4
        elif mantissa <= 72:
            mantissa = 4 << 4
        elif mantissa <= 87:
            mantissa = 5 << 4
        elif mantissa <= 104:
            mantissa = 6 << 4
        elif mantissa <= 119:
            mantissa = 7 << 4
        else:
            mantissa = 0
            exponent += 1
            if exponent == 255:
                exponent = 254
                mantissa = 7 << 4
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16)
    float_value = int_to_float(bits)
    if negative:
        float_value = float_value * -1
    return float_value

class FakeQuantizeE8M1Tests(unittest.TestCase):
    def SetUp(self):
        torch.manual_seed(0)

    def create_2d_matrix_uint32(self, dim_x, dim_y):
        int_tensor = torch.ones(dim_x, dim_y, dtype=torch.uint32)
        # The following number will be (1.0, -1.0) in bf16
        default_value = (16256 * 2 ** 16 + 49024)
        # Gradually increase from (1.0, -1.0) to slightly larger values.
        for i in range(dim_x):
            for j in range(dim_y):
                int_tensor[i][j] = default_value + i * (2 ** 16) + j
        return int_tensor

    def create_2d_random_matrix_bf16(self, dim_x, dim_y, range_start, range_end):
        return torch.FloatTensor(dim_x, dim_y).uniform_(range_start, range_end).type(torch.bfloat16)

    def compute_total_relative_error(self, tensor, dim_x, dim_y, block_x, block_y, global_mantissa):
        total_relative_error = 0
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                block = tensor[x : x + block_x, y : y + block_y].float()
                max_value = torch.max(torch.abs(block))
                block_sf = 448.0 / max_value.item()
                sign, exponent, mantissa = decompose_float(block_sf)
                new_block_sf = sign << 31 | exponent << 23 | global_mantissa
                new_block_sf = int_to_float(new_block_sf)
                if new_block_sf <= block_sf:
                    scaling_factor = new_block_sf
                else:
                    scaling_factor = new_block_sf / 2.0

                block_scaled = block * scaling_factor
                quantized = block_scaled.type(torch.float8_e4m3fn)
                dequantized = quantized.type(torch.float32)
                block_descaled = dequantized / scaling_factor

                relative_error = torch.abs((block_descaled - block) / block)
                total_relative_error += torch.sum(relative_error)
        return total_relative_error

    def fake_quantize_e8mx(self, tensor, dim_x, dim_y, block_x, block_y, quantize_type):
        output_tensor = tensor.clone()
        total_relative_error = 0
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                block = output_tensor[x : x + block_x, y : y + block_y].float()
                for i in range(0, block_x):
                    for j in range(0, block_y):
                        element = block[i, j].item()
                        if quantize_type == 1:
                            new_element = e8m1_quant(element)
                        elif quantize_type == 2:
                            new_element = e8m2_quant(element)
                        elif quantize_type == 3:
                            new_element = e8m3_quant(element)
                        block[i, j] = new_element

                block_bf16 = block.type(torch.bfloat16)
                output_tensor[x : x + block_x, y : y + block_y] = block_bf16
        return output_tensor


    def test_uint32_matrix_bf16_e8m1(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out = mor.ops.fake_quantize_e8mx(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            quantize_type=1,
            scaling_strategy=3,
        )

        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=1)

        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(bf16_mor_out.cpu()))

    def test_uint32_matrix_bf16_inplace_e8m1(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)
        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=1)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        mor.ops.fake_quantize_e8mx_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            1,
            3,
            matrix_cuda)


        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(matrix_cuda.cpu()))


    def test_uint32_matrix_bf16_e8m2(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out = mor.ops.fake_quantize_e8mx(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            quantize_type=2,
            scaling_strategy=3,
        )

        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=2)

        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(bf16_mor_out.cpu()))

    def test_uint32_matrix_bf16_inplace_e8m2(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)
        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=2)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        mor.ops.fake_quantize_e8mx_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            2,
            3,
            matrix_cuda)


        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(matrix_cuda.cpu()))

    def test_uint32_matrix_bf16_e8m3(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out = mor.ops.fake_quantize_e8mx(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            quantize_type=3,
            scaling_strategy=3,
        )

        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=3)

        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(bf16_mor_out.cpu()))

    def test_uint32_matrix_bf16_inplace_e8m3(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)
        cpu_quantized = self.fake_quantize_e8mx(matrix_bf16, dim_x, dim_y, block_x, block_y, quantize_type=3)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        mor.ops.fake_quantize_e8mx_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            3,
            3,
            matrix_cuda)


        # mor_cpu = bf16_mor_out.cpu()
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_cpu[x, y] != cpu_quantized[x, y]:
        #             print(f"[{x}, {y}]: original = {matrix_bf16[x, y]}, mor = {mor_cpu[x, y]}, cpu = {cpu_quantized[x, y]}")

        self.assertTrue(cpu_quantized.equal(matrix_cuda.cpu()))


    def test_uint32_matrix_bf16_e8m3_global_amax(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out = mor.ops.fake_quantize_e8mx(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            quantize_type=1,
            scaling_strategy=1,
        )

        mor.ops.fake_quantize_e8mx_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            1,
            1,
            matrix_cuda)
        self.assertTrue(bf16_mor_out.equal(matrix_cuda))

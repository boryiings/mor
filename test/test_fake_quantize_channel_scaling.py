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
    mantissa = bin_value & 0x7fffff
    return sign, exponent, mantissa

class FakeQuantizeChannelScalingTests(unittest.TestCase):
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

    def fake_quantize_e4m3(self, tensor, dim_x, dim_y, block_x, block_y, global_mantissa):
        output_tensor = tensor.clone()
        total_relative_error = 0
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                block = output_tensor[x : x + block_x, y : y + block_y].float()
                max_value = torch.max(torch.abs(block))
                block_sf = 448.0 / max_value.item()
                sign, exponent, mantissa = decompose_float(block_sf)
                new_block_sf = sign << 31 | exponent << 23 | global_mantissa
                new_block_sf = int_to_float(new_block_sf)
                _, _, local_mantissa = decompose_float(block_sf)
                if global_mantissa <= local_mantissa:
                    scaling_factor = new_block_sf
                else:
                    scaling_factor = new_block_sf / 2.0

                block_scaled = block * scaling_factor
                quantized = block_scaled.type(torch.float8_e4m3fn)
                dequantized = quantized.type(torch.float32)
                block_descaled = dequantized / scaling_factor
                block_bf16 = block_descaled.type(torch.bfloat16)
                output_tensor[x : x + block_x, y : y + block_y] = block_bf16
        return output_tensor

    def test_uint32_matrix_e4m3_row(self):
        dim_x = 256
        dim_y = 256
        block_x = 1
        block_y = 256
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)
        quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        self.assertTrue(quantized_tensor.equal(bf16_mor_out))

    def test_uint32_matrix_e4m3_col(self):
        dim_x = 256
        dim_y = 256
        block_x = 256
        block_y = 1
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)
        quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        self.assertTrue(quantized_tensor.equal(bf16_mor_out))


    def test_uint32_matrix_bf16_row(self):
        dim_x = 256
        dim_y = 256
        block_x = 1
        block_y = 256
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)

    def test_uint32_matrix_bf16_col(self):
        dim_x = 256
        dim_y = 256
        block_x = 256
        block_y = 1
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the tensor.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)

    def test_random_matrix_large_block_small_range_row(self):
        dim_x = 256
        dim_y = 256
        block_x = 1
        block_y = 256
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -448.0, 448.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Small range Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Small range PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Small range Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)
        quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)

        # mor_cpu = bf16_mor_out.to("cpu")
        # quantized_cpu = quantized_tensor.to("cpu")
        # for x in range(0, dim_x):
        #     for y in range(0, dim_y):
        #         if quantized_cpu[x][y] != mor_cpu[x][y]:
        #             print(f"Error at [{x}][{y}]")
        #             print(f"PyTorch value = {quantized_tensor[x][y]}")
        #             print(f"MoR value = {mor_cpu[x][y]}")
        self.assertTrue(quantized_tensor.equal(bf16_mor_out))

    def test_random_matrix_large_block_small_range_col(self):
        dim_x = 256
        dim_y = 256
        block_x = 256
        block_y = 1
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -448.0, 448.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Small range Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Small range PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Small range Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-03, atol=0)
        quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)

        # mor_cpu = bf16_mor_out.to("cpu")
        # quantized_cpu = quantized_tensor.to("cpu")
        # for x in range(0, dim_x):
        #     for y in range(0, dim_y):
        #         if quantized_cpu[x][y] != mor_cpu[x][y]:
        #             print(f"Error at [{x}][{y}]")
        #             print(f"PyTorch value = {quantized_tensor[x][y]}")
        #             print(f"MoR value = {mor_cpu[x][y]}")
        self.assertTrue(quantized_tensor.equal(bf16_mor_out))

    def test_random_matrix_large_block_large_range_row(self):
        dim_x = 256
        dim_y = 256
        block_x = 1
        block_y = 256
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Large range Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Large range PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        if global_error.item() / global_non_zero.item() > 0.045:
            self.assertTrue(meta.item() == 3)
        else:
            self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Large range Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-02, atol=0)
        if meta.item() == 1:
            quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
            # mor_cpu = bf16_mor_out.to("cpu")
            # quantized_cpu = quantized_tensor.to("cpu")
            # for x in range(0, dim_x):
            #     for y in range(0, dim_y):
            #         if quantized_cpu[x][y] != mor_cpu[x][y]:
            #             print(f"Error at [{x}][{y}]")
            #             print(f"PyTorch value = {quantized_cpu[x][y]}")
            #             print(f"MoR value = {mor_cpu[x][y]}")

            self.assertTrue(quantized_tensor.equal(bf16_mor_out))
        else:
            self.assertTrue(matrix_cuda.equal(bf16_mor_out))

    def test_random_matrix_large_block_large_range_col(self):
        dim_x = 256
        dim_y = 256
        block_x = 256
        block_y = 1
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            e4m3_threshold=0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_cuda))
        torch_non_zero = torch.nonzero(matrix_cuda).size(0)
        print(f"Large range Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Large range PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        if global_error.item() / global_non_zero.item() > 0.045:
            self.assertTrue(meta.item() == 3)
        else:
            self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Large range Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-02, atol=0)
        if meta.item() == 1:
            quantized_tensor = self.fake_quantize_e4m3(matrix_cuda, dim_x, dim_y, block_x, block_y, global_mantissa)
            # mor_cpu = bf16_mor_out.to("cpu")
            # quantized_cpu = quantized_tensor.to("cpu")
            # for x in range(0, dim_x):
            #     for y in range(0, dim_y):
            #         if quantized_cpu[x][y] != mor_cpu[x][y]:
            #             print(f"Error at [{x}][{y}]")
            #             print(f"PyTorch value = {quantized_cpu[x][y]}")
            #             print(f"MoR value = {mor_cpu[x][y]}")

            self.assertTrue(quantized_tensor.equal(bf16_mor_out))
        else:
            self.assertTrue(matrix_cuda.equal(bf16_mor_out))

    def test_random_matrix_large_block_large_range_inplace_row(self):
        dim_x = 256
        dim_y = 256
        block_x = 1
        block_y = 256
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        matrix_copy = matrix_cuda.clone()
        meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_copy))
        torch_non_zero = torch.nonzero(matrix_copy).size(0)
        print(f"Inplace Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Inplace PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        if global_error.item() / global_non_zero.item() > 0.045:
            self.assertTrue(meta.item() == 3)
        else:
            self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_copy, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Inplace Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-02, atol=0)
        if meta.item() == 1:
            quantized_tensor = self.fake_quantize_e4m3(matrix_copy, dim_x, dim_y, block_x, block_y, global_mantissa)
            # mor_cpu = matrix_cuda.to("cpu")
            # quantized_cpu = quantized_tensor.to("cpu")
            # for x in range(0, dim_x):
            #     for y in range(0, dim_y):
            #         if quantized_cpu[x][y] != mor_cpu[x][y]:
            #             print(f"Error at [{x}][{y}]")
            #             print(f"PyTorch value = {quantized_cpu[x][y]}")
            #             print(f"MoR value = {mor_cpu[x][y]}")

            self.assertTrue(quantized_tensor.equal(matrix_cuda))
        else:
            self.assertTrue(matrix_copy.equal(matrix_cuda))

    def test_random_matrix_large_block_large_range_inplace_col(self):
        dim_x = 256
        dim_y = 256
        block_x = 256
        block_y = 1
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        matrix_copy = matrix_cuda.clone()
        meta, global_amax, global_non_zero, global_error = mor.ops.fake_quantize_channel_scaling_inplace(
            matrix_cuda,
            dim_x,
            dim_y,
            block_x,
            block_y,
            0.045,
            ss_mode=1)

        torch_amax = torch.max(torch.abs(matrix_copy))
        torch_non_zero = torch.nonzero(matrix_copy).size(0)
        print(f"Inplace Meta = {meta}, global amax = {global_amax}, global non zero = {global_non_zero}, global error = {global_error}")
        print(f"Inplace PyTorch amax = {torch_amax}, PyTorch non zero = {torch_non_zero}")
        if global_error.item() / global_non_zero.item() > 0.045:
            self.assertTrue(meta.item() == 3)
        else:
            self.assertTrue(meta.item() == 1)
        self.assertTrue(global_amax.item() == torch_amax.item())
        self.assertTrue(global_non_zero.item() == torch_non_zero)

        global_sf = 448.0 / torch_amax.item()
        _, _, global_mantissa = decompose_float(global_sf)
        total_relative_error = self.compute_total_relative_error(matrix_copy, dim_x, dim_y, block_x, block_y, global_mantissa)
        print(f"Inplace Total relative error = {total_relative_error}")
        torch.testing.assert_close(total_relative_error, global_error[0], rtol=1e-02, atol=0)
        if meta.item() == 1:
            quantized_tensor = self.fake_quantize_e4m3(matrix_copy, dim_x, dim_y, block_x, block_y, global_mantissa)
            # mor_cpu = matrix_cuda.to("cpu")
            # quantized_cpu = quantized_tensor.to("cpu")
            # for x in range(0, dim_x):
            #     for y in range(0, dim_y):
            #         if quantized_cpu[x][y] != mor_cpu[x][y]:
            #             print(f"Error at [{x}][{y}]")
            #             print(f"PyTorch value = {quantized_cpu[x][y]}")
            #             print(f"MoR value = {mor_cpu[x][y]}")

            self.assertTrue(quantized_tensor.equal(matrix_cuda))
        else:
            self.assertTrue(matrix_copy.equal(matrix_cuda))

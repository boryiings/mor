# Python imports
import math
import unittest

# Library imports
import numpy as np
import torch

# Project imports
import mor

class FakeQuantizeTests(unittest.TestCase):
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

    def create_2d_random_matrix_fp32(self, dim_x, dim_y, range_start, range_end):
        return torch.FloatTensor(dim_x, dim_y).uniform_(range_start, range_end).type(torch.float)


    def fake_quantize_to_fp32(self, block, dtype, amax, use_e8_scaling = False):
        if dtype == torch.float8_e4m3fn:
            scale = 448.0 / amax
        elif dtype == torch.float8_e5m2:
            scale = 57344.0 / amax
        elif dtype == "f6_e2m3":
            scale = 7.5 / amax
            dtype = torch.float8_e4m3fn
        if use_e8_scaling:
            scale = 2.0 ** math.floor(math.log2(scale))
        block_scaled = block * scale
        quantized = block_scaled.type(dtype)
        dequantized = quantized.float()
        block_descaled = dequantized / scale
        return block_descaled

    def relative_error(self, original, quantized):
        self.assertTrue(torch.all(original != 0))
        diff = quantized - original
        relative_error = diff / original
        abs_error = torch.sum(torch.abs(relative_error))
        return abs_error

    def cpu_mor_dynamic_range(self, block, use_e8_scaling=False):
        self.assertEqual(block.device.type, "cpu")
        # Compute amax/amin/dynamic range in float.
        block_float = block.float()
        block_abs = torch.abs(block_float)
        amax = torch.max(block_abs).item()
        non_zero = block_abs[block_abs > 0]
        if torch.numel(non_zero) > 0:
            amin = torch.min(non_zero)
            dyn_range = amax / amin
        else:
            amin = 0.0
            dyn_range = 1
            return block

        # Do MoR
        if dyn_range <= 7.5:
            # E2M3
            block_descaled = self.fake_quantize_to_fp32(block_float, "f6_e2m3", amax, use_e8_scaling)
            block_bf16 = block_descaled.type(torch.bfloat16)
            return block_bf16
        elif dyn_range <= 28762:
            # E4M3
            block_descaled = self.fake_quantize_to_fp32(block_float, torch.float8_e4m3fn, amax, use_e8_scaling)
            block_bf16 = block_descaled.type(torch.bfloat16)
            return block_bf16
        elif dyn_range <= 939524096:
            # E5M2
            block_descaled = self.fake_quantize_to_fp32(block_float, torch.float8_e5m2, amax, use_e8_scaling)
            block_bf16 = block_descaled.type(torch.bfloat16)
            return block_bf16
        else:
            # bf16
            return block

    def cpu_mor(self, block):
        self.assertEqual(block.device.type, "cpu")
        # Compute amax/amin/dynamic range in float.
        block_float = block.float()
        block_abs = torch.abs(block_float)
        # Found an interesting difference here.
        # If amax is a tensor, then the scale value
        # of 448.0 / 240.0 will be 1072623344 in uint32.
        # If amax is a scalar (with the .item() operation), then
        # the scale value of 448.0 / 240.0 will be 1072623343 in uint32.
        # There is one bit difference between the two.
        # If we remove the item() call, the test will actually fail
        # because of this one-bit difference.
        amax = torch.max(block_abs).item()
        non_zero = block_abs[block_abs > 0]
        if torch.numel(non_zero) > 0:
            amin = torch.min(non_zero)
            dyn_range = amax / amin
        else:
            amin = 0.0
            dyn_range = 1

        # Do MoR
        if dyn_range <= 7.5:
            # E2M3
            block_descaled = self.fake_quantize_to_fp32(block_float, "f6_e2m3", amax)
            block_bf16 = block_descaled.type(torch.bfloat16)
            return block_bf16
        elif dyn_range <= 28762 * 4:
            # E4M3
            block_descaled = self.fake_quantize_to_fp32(block_float, torch.float8_e4m3fn, amax)
            block_bf16 = block_descaled.type(torch.bfloat16)
            return block_bf16
        elif dyn_range <= 939524096:
            # E5M2
            # Step 1: Calculate the E4M3 error
            e4m3_block_descaled = self.fake_quantize_to_fp32(block_float, torch.float8_e4m3fn, amax)
            e4m3_error = self.relative_error(block_float, e4m3_block_descaled)
            # Step 2: Calculate the E5M2 error
            e5m2_block_descaled = self.fake_quantize_to_fp32(block_float, torch.float8_e5m2, amax)
            e5m2_error = self.relative_error(block_float, e5m2_block_descaled)

            print(f"e4m3_error = {e4m3_error}, e5m2_error = {e5m2_error}")
            if e4m3_error <= e5m2_error:
                block_bf16 = e4m3_block_descaled.type(torch.bfloat16)
                return block_bf16
            # MoR V3 promotes all e5m2 blocks to bf16
            # else:
            #     block_bf16 = e5m2_block_descaled.type(torch.bfloat16)
            return block
        else:
            # bf16
            return block

    def test_uint32_matrix_small_block(self):
        # MoR does not work for thread block < 32 for now. Skip.
        return
        dim_x = 16
        dim_y = 16
        block_x = 4
        block_y = 4
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (38912, -38912) in bf16.
        # Will force to use e5m2 for the block.
        matrix_uint32[4][1] = (18200 * 2 ** 16 + 50968)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[8][1] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=0.0001,
                                                   mode=1,
                                                   sf_type=1)

        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                if i == 1 and j == 0:
                    # E5M2 case.
                    self.assertEqual(meta_cpu[i][j], 2)
                elif i == 2 and j == 0:
                    # bf16 case.
                    self.assertEqual(meta_cpu[i][j], 3)
                else:
                    # E4M3 case.
                    self.assertEqual(meta_cpu[i][j], 1)
        
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_random_matrix_small_block(self):
        # MoR does not work for thread block < 32 for now. Skip.
        return
        dim_x = 16
        dim_y = 16
        block_x = 4
        block_y = 4
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, _ = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=0.0001,
                                                mode=1,
                                                sf_type=1)
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_uint32_matrix_large_block_dynamic_range(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (38912, -38912) in bf16.
        # Will force to use e5m2 for the block.
        matrix_uint32[128][1] = (18200 * 2 ** 16 + 50968)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)
        # The following will be (238, -238) in bf16.
        # Will force to use e4m3 for the block.
        matrix_uint32[1][1] = (17262 * 2 ** 16 + 50030)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-15,
                                                   mode=101,
                                                   sf_type=1)
        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                if i == 1 and j == 0:
                    # E5M2 case.
                    self.assertEqual(meta_cpu[i][j], 2)
                elif i == 0 and j == 1:
                    # bf16 case.
                    self.assertEqual(meta_cpu[i][j], 3)
                elif i == 1 and j == 1:
                    # E2M3 case.
                    self.assertEqual(meta_cpu[i][j], 4)
                else:
                    # E4M3 case.
                    self.assertEqual(meta_cpu[i][j], 1)
        
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # for i in range(0, block_x):
                #     for j in range(0, block_y):
                #         if bf16_mor_block[i][j] != cpu_block_mor[i][j]:
                #             print(f"x = {x}, y = {y}, original = {cpu_block[i][j]}, mor gpu = {bf16_mor_block[i][j]}, mor cpu = {cpu_block_mor[i][j]}")
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_random_matrix_large_block_dynamic_range(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, _ = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=1e-15,
                                                mode=101,
                                                sf_type=1)
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_uint32_matrix_large_block(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (116224, -116224) in bf16.
        # This is in the E5M2 range. Even though there is only one value, 
        # this value will contribute to 100% quantization error.
        # So the total error is still larger in E4M3. Thus this value
        # forces MoR to choose E5M2.
        matrix_uint32[128][1] = (18403 * 2 ** 16 + 51171)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)
        # The following will be (238, -238) in bf16.
        # Will force to use e4m3 for the block.
        matrix_uint32[1][1] = (17264 * 2 ** 16 + 50032)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-15,
                                                   mode=1,
                                                   sf_type=1)
        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                if i == 1 and j == 0:
                    # E5M2 case.
                    self.assertEqual(meta_cpu[i][j], 3)
                elif i == 0 and j == 1:
                    # bf16 case.
                    self.assertEqual(meta_cpu[i][j], 3)
                elif i == 1 and j == 1:
                    # E2M3 case.
                    self.assertEqual(meta_cpu[i][j], 4)
                else:
                    # E4M3 case.
                    self.assertEqual(meta_cpu[i][j], 1)
        
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # for i in range(0, block_x):
                #     for j in range(0, block_y):
                #         if bf16_mor_block[i][j] != cpu_block_mor[i][j]:
                #             print(f"x = {i}, y = {j}, uint32 = {matrix_uint32[i][j // 2]}, original = {cpu_block[i][j]}, mor gpu = {bf16_mor_block[i][j]}, mor cpu = {cpu_block_mor[i][j]}")

                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_random_matrix_large_block(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=1e-15,
                                                mode=1,
                                                sf_type=1)
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor(cpu_block) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                if bf16_mor_block.equal(cpu_block_mor) is False:
                    print(f"x = {x}, y = {y}")
                    print("MoR block")
                    print(bf16_mor_block)
                    print("cpu mor block")
                    print(cpu_block_mor)
                    print("original cpu block")
                    print(cpu_block)
                    print("original amax", torch.max(torch.abs(cpu_block)))
                    print("meta = ", meta)
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_random_matrix_large_block_current_scaling_e4m3(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        matrix_amax = torch.max(torch.abs(matrix_cuda)).float()
        scaling_factor = 448.0 / matrix_amax
        bf16_mor_out, _ = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=scaling_factor,
                                                mode=4,
                                                sf_type=1)
        # CPU current scaling.
        cpu_current_scaling = self.fake_quantize_to_fp32(matrix_bf16.float(), torch.float8_e4m3fn, matrix_amax.cpu())
        cpu_current_scaling = cpu_current_scaling.type(torch.bfloat16)
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        self.assertTrue(bf16_mor_cpu.equal(cpu_current_scaling))

    def test_random_matrix_large_block_current_scaling_e5m2(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        matrix_amax = torch.max(torch.abs(matrix_cuda)).float()
        scaling_factor = 57344.0 / matrix_amax
        bf16_mor_out, _ = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=scaling_factor,
                                                mode=5,
                                                sf_type=1)
        # CPU current scaling.
        cpu_current_scaling = self.fake_quantize_to_fp32(matrix_bf16.float(), torch.float8_e5m2, matrix_amax.cpu())
        cpu_current_scaling = cpu_current_scaling.type(torch.bfloat16)
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        self.assertTrue(bf16_mor_cpu.equal(cpu_current_scaling))

    def test_random_matrix_bf16_block(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-15,
                                                   mode=7,
                                                   sf_type=1)

        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                # bf16 case.
                self.assertEqual(meta_cpu[i][j], 3)

        # Check if the MoR tensor is exactly the same as the original bf16 tensor.
        self.assertTrue(matrix_cuda.equal(bf16_mor_out))

    def test_no_alloc(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (38912, -38912) in bf16.
        # Will force to use e5m2 for the block.
        matrix_uint32[128][1] = (18200 * 2 ** 16 + 50968)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        output = torch.zeros(dim_x, dim_y, dtype=torch.bfloat16, device="cuda")
        meta_no_alloc = mor.ops.fake_quantize_no_alloc(matrix_cuda,
                                                       dim_x,
                                                       dim_y,
                                                       block_x,
                                                       block_y,
                                                       clamp_threshold=1e-15,
                                                       mode=1,
                                                       sf_type=1,
                                                       output=output)
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-15,
                                                   mode=1,
                                                   sf_type=1)
        self.assertTrue(meta_no_alloc.equal(meta))
        self.assertTrue(output.equal(bf16_mor_out))

    def test_inplace(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (38912, -38912) in bf16.
        # Will force to use e5m2 for the block.
        matrix_uint32[128][1] = (18200 * 2 ** 16 + 50968)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        output = torch.zeros(dim_x, dim_y, dtype=torch.bfloat16, device="cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-15,
                                                   mode=1,
                                                   sf_type=1)
        meta_no_alloc = mor.ops.fake_quantize_no_alloc(matrix_cuda,
                                                       dim_x,
                                                       dim_y,
                                                       block_x,
                                                       block_y,
                                                       clamp_threshold=1e-15,
                                                       mode=1,
                                                       sf_type=1,
                                                       output=matrix_cuda)
        self.assertTrue(meta_no_alloc.equal(meta))
        self.assertTrue(matrix_cuda.equal(bf16_mor_out))


    def test_quant_dequant_e4m3(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -400.0, 400.0)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e4m3(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_simulated_matrix = matrix_fp32.type(torch.float8_e4m3fn).type(torch.float)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_matrix_cpu[x, y] != cpu_simulated_matrix[x, y]:
        #             print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {cpu_simulated_matrix[x, y]}")
        self.assertTrue(mor_matrix_cpu.equal(cpu_simulated_matrix))

    def test_quant_dequant_subnormal(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -1.0, 1.0)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e4m3(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_simulated_matrix = matrix_fp32.type(torch.float8_e4m3fn).type(torch.float)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_matrix_cpu[x, y] != cpu_simulated_matrix[x, y]:
        #             print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {cpu_simulated_matrix[x, y]}")
        self.assertTrue(mor_matrix_cpu.equal(cpu_simulated_matrix))

    def test_quant_dequant_e4m3_nan(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -1000.0, 1000.0)
        matrix_fp32[0, 0] = 463
        matrix_fp32[0, 1] = 464
        matrix_fp32[0, 2] = 465
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e4m3(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_simulated_matrix = matrix_fp32.type(torch.float8_e4m3fn).type(torch.float)
        for x in range(dim_x):
            for y in range(dim_y):
                mor_isnan = math.isnan(mor_matrix_cpu[x, y])
                cpu_isnan = math.isnan(cpu_simulated_matrix[x, y])
                if mor_isnan != cpu_isnan:
                    print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {cpu_simulated_matrix[x, y]}")
                self.assertTrue(mor_isnan == cpu_isnan)
                if not mor_isnan:
                    self.assertTrue(mor_matrix_cpu[x, y] == cpu_simulated_matrix[x, y])


    def test_quant_dequant_e5m2(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -53768.0, 53768.0)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e5m2(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_simulated_matrix = matrix_fp32.type(torch.float8_e5m2).type(torch.float)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_matrix_cpu[x, y] != cpu_simulated_matrix[x, y]:
        #             print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {cpu_simulated_matrix[x, y]}")
        self.assertTrue(mor_matrix_cpu.equal(cpu_simulated_matrix))

    def cpu_e3m2_quant_dequant(self, cpu_tensor):
        quantized_tensor = cpu_tensor.type(torch.float8_e5m2).type(torch.float)
        fix_overflow = torch.where(quantized_tensor >= 28.0, 28.0, quantized_tensor)
        fix_underflow = torch.where(fix_overflow <= -28.0, -28.0, fix_overflow)
        return fix_underflow

    def test_quant_dequant_e3m2(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, 0.25, 100.0)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e3m2(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_quant_dequant_tensor = self.cpu_e3m2_quant_dequant(matrix_fp32)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_matrix_cpu[x][y] != cpu_quant_dequant_tensor[x][y]:
        #             print(f"Original value: {matrix_fp32[x][y]}, mor value = {mor_matrix_cpu[x][y]}, cpu value = {cpu_quant_dequant_tensor[x][y]}")
        self.assertTrue(mor_matrix_cpu.equal(cpu_quant_dequant_tensor))


    def cpu_e2m3_quant_dequant(self, cpu_tensor):
        quantized_tensor = cpu_tensor.type(torch.float8_e4m3fn).type(torch.float)
        fix_overflow = torch.where(quantized_tensor >= 7.5, 7.5, quantized_tensor)
        fix_underflow = torch.where(fix_overflow <= -7.5, -7.5, fix_overflow)
        return fix_underflow

    def test_quant_dequant_e2m3(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, 1.0, 20.0)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.quant_dequant_e2m3(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        cpu_quant_dequant_tensor = self.cpu_e2m3_quant_dequant(matrix_fp32)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if mor_matrix_cpu[x][y] != cpu_quant_dequant_tensor[x][y]:
        #             print(f"Original value: {matrix_fp32[x][y]}, mor value = {mor_matrix_cpu[x][y]}, cpu value = {cpu_quant_dequant_tensor[x][y]}")
        self.assertTrue(mor_matrix_cpu.equal(cpu_quant_dequant_tensor))


    def e8_scale_rne_cpu(self, value_fp32):
        value_abs = torch.abs(value_fp32)
        scale_fp32 = 448.0 / value_abs
        smaller_exp = 2 ** torch.floor(torch.log2(scale_fp32))
        larger_exp = smaller_exp * 2

        # Find the scale factor that is closer.
        scaled_value_small = value_abs * smaller_exp
        scaled_value_large = value_abs * larger_exp
        small_diff = (value_abs - scaled_value_small) / value_abs
        large_diff = (scaled_value_large - value_abs) / value_abs
        e8_scale = torch.where(small_diff <= large_diff, smaller_exp, larger_exp)
        e8_scale = torch.where(scaled_value_large <= 464, e8_scale, smaller_exp)
        scaled_value = value_fp32 * e8_scale

        # Do quantize and dequantize
        dequantized = scaled_value.type(torch.float8_e4m3fn).type(torch.float)
        descaled = dequantized / e8_scale
        return descaled

    def e8_scale_rz_cpu(self, value_fp32):
        value_abs = torch.abs(value_fp32)
        scale_fp32 = 448.0 / value_abs
        e8_scale = 2 ** torch.floor(torch.log2(scale_fp32))

        # Find the scale factor that is closer.
        scaled_value = value_fp32 * e8_scale

        # Do quantize and dequantize
        dequantized = scaled_value.type(torch.float8_e4m3fn).type(torch.float)
        descaled = dequantized / e8_scale
        return descaled



    def test_e4m3_with_e8_scale_rne(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -1000, 1000)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.e4m3_with_e8_scale_rne(matrix_cuda,
                                                    dim_x,
                                                    dim_y,
                                                    block_x,
                                                    block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        pruned_matrix_cpu = self.e8_scale_rne_cpu(matrix_fp32)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if pruned_matrix_cpu[x, y] != mor_matrix_cpu[x, y]:
        #             print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {pruned_matrix_cpu[x, y]}")
        self.assertTrue(mor_matrix_cpu.equal(pruned_matrix_cpu))

    def test_e4m3_with_e8_scale_rz(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_fp32 = self.create_2d_random_matrix_fp32(dim_x, dim_y, -1000, 1000)
        matrix_cuda = matrix_fp32.to("cuda")
        mor_matrix = mor.ops.e4m3_with_e8_scale_rz(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y)
        mor_matrix_cpu = mor_matrix.to("cpu")
        pruned_matrix_cpu = self.e8_scale_rz_cpu(matrix_fp32)
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         if pruned_matrix_cpu[x, y] != mor_matrix_cpu[x, y]:
        #             print(f"Original value: {matrix_fp32[x, y]}, mor value = {mor_matrix_cpu[x, y]}, cpu value = {pruned_matrix_cpu[x, y]}")
        self.assertTrue(mor_matrix_cpu.equal(pruned_matrix_cpu))

    def test_uint32_matrix_large_block_dynamic_range_e8_scaling(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_uint32 = self.create_2d_matrix_uint32(dim_x, dim_y // 2)
        # The following will be (38912, -38912) in bf16.
        # Will force to use e5m2 for the block.
        matrix_uint32[128][1] = (18200 * 2 ** 16 + 50968)
        # The following will be (1.4603e+11, -1.4603e+11) in bf16.
        # Will force to use bf16 for the block.
        matrix_uint32[0][70] = (21000 * 2 ** 16 + 53768)
        # The following will be (238, -238) in bf16.
        # Will force to use e4m3 for the block.
        matrix_uint32[1][1] = (17262 * 2 ** 16 + 50030)

        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = matrix_uint32.view(dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-25,
                                                   mode=101,
                                                   sf_type=2)
        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                if i == 1 and j == 0:
                    # E5M2 case.
                    self.assertEqual(meta_cpu[i][j], 2)
                elif i == 0 and j == 1:
                    # bf16 case.
                    self.assertEqual(meta_cpu[i][j], 3)
                elif i == 1 and j == 1:
                    # E2M3 case.
                    self.assertEqual(meta_cpu[i][j], 4)
                else:
                    # E4M3 case.
                    self.assertEqual(meta_cpu[i][j], 1)
        
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block, use_e8_scaling=True) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # for i in range(0, block_x):
                #     for j in range(0, block_y):
                #         if bf16_mor_block[i][j] != cpu_block_mor[i][j]:
                #             print(f"x = {x}, {i}, y = {y}, {j}, original = {cpu_block[i][j]}, mor gpu = {bf16_mor_block[i][j]}, mor cpu = {cpu_block_mor[i][j]}")
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_random_matrix_large_block_dynamic_range_e8_scaling(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        matrix_bf16 = self.create_2d_random_matrix_bf16(dim_x, dim_y, -53768.0, 53768.0)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, _ = mor.ops.fake_quantize(matrix_cuda,
                                                dim_x,
                                                dim_y,
                                                block_x,
                                                block_y,
                                                clamp_threshold=1e-25,
                                                mode=101,
                                                sf_type=2)
        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block, use_e8_scaling=True) 
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_uint32_matrix_zero_block(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = torch.zeros(dim_x, dim_y, dtype=torch.bfloat16)

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-25,
                                                   mode=101,
                                                   sf_type=2)
        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                self.assertEqual(meta_cpu[i][j], 4)

        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block, use_e8_scaling=True)
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # for i in range(0, block_x):
                #     for j in range(0, block_y):
                #         if bf16_mor_block[i][j] != cpu_block_mor[i][j]:
                #             print(f"x = {x}, {i}, y = {y}, {j}, original = {cpu_block[i][j]}, mor gpu = {bf16_mor_block[i][j]}, mor cpu = {cpu_block_mor[i][j]}")
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

    def test_uint32_matrix_small_value_block(self):
        dim_x = 256
        dim_y = 256
        block_x = 128
        block_y = 128
        # Reinterpret the uint32 integers into bf16 values.
        matrix_bf16 = torch.ones(dim_x, dim_y, dtype=torch.bfloat16) * 1e-27

        # Call MoR CUDA kernel.
        matrix_cuda = matrix_bf16.to("cuda")
        bf16_mor_out, meta = mor.ops.fake_quantize(matrix_cuda,
                                                   dim_x,
                                                   dim_y,
                                                   block_x,
                                                   block_y,
                                                   clamp_threshold=1e-25,
                                                   mode=101,
                                                   sf_type=2)
        # Check meta
        meta_cpu = meta.to("cpu")
        for i in range(0, dim_x // block_x):
            for j in range(0, dim_y // block_y):
                self.assertEqual(meta_cpu[i][j], 4)

        # Loop over each block
        bf16_mor_cpu = bf16_mor_out.to("cpu")
        for x in range(0, dim_x, block_x):
            for y in range(0, dim_y, block_y):
                print(f"Checking block [{x // block_x}][{y // block_y}]")
                cpu_block = matrix_bf16[x : x + block_x, y : y + block_y]
                cpu_block_mor = self.cpu_mor_dynamic_range(cpu_block, use_e8_scaling=True)
                bf16_mor_block = bf16_mor_cpu[x : x + block_x, y : y + block_y]
                # for i in range(0, block_x):
                #     for j in range(0, block_y):
                #         if bf16_mor_block[i][j] != cpu_block_mor[i][j]:
                #             print(f"x = {x}, {i}, y = {y}, {j}, original = {cpu_block[i][j]}, mor gpu = {bf16_mor_block[i][j]}, mor cpu = {cpu_block_mor[i][j]}")
                self.assertTrue(bf16_mor_block.equal(cpu_block_mor))

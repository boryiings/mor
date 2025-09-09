#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>


namespace cg = cooperative_groups;
__global__ void combinedKernelPhases1And2(
        const uint32_t* d_A_packed, uint32_t *d_B_packed,
        int M_packed, int N_packed,
        int block_ROWS_packed, int block_COLS_packed,
        int sub_tensors_x, int sub_tensors_y,
        int* d_block_non_zero,      int* d_global_non_zero,
        int* d_block_flush_to_zero, int* d_global_flush_to_zero,
        float* d_block_amax,        float* d_global_amax,
        float* d_block_error,       float* d_global_error,
        int *d_quant_type_used
        );
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


__global__ void compute_E8M1_Representation(
        uint32_t* d_A_packed, uint32_t* d_B_packed,
        int M_packed, int N_packed,
        int block_ROWS_packed, int block_COLS_packed,
        int sub_tensors_x, int sub_tensors_y)
{

    int total_sub_tensors = sub_tensors_x * sub_tensors_y;

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    //int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int num_blocks = gridDim.x * gridDim.y;

    int sub_tensors_per_block = (total_sub_tensors + num_blocks - 1) / num_blocks;

    int start_sub_tensor_id = (block_id + 0) * sub_tensors_per_block;
    int   end_sub_tensor_id = (block_id + 1) * sub_tensors_per_block;

    if (start_sub_tensor_id > total_sub_tensors) start_sub_tensor_id = total_sub_tensors;
    if (  end_sub_tensor_id > total_sub_tensors)   end_sub_tensor_id = total_sub_tensors;

    //if ((block_id == 0) && (tid == 0)) printf("sub_tensors_per_block = %d :: start_sub_tensor_id = %d :: end_sub_tensor_id = %d\n", sub_tensors_per_block, start_sub_tensor_id, end_sub_tensor_id);

    // Iterate through the assigned sub-tensors for this block
    for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id)
    {
        int sub_tensor_row = sub_tensor_id / sub_tensors_x;
        int sub_tensor_col = sub_tensor_id % sub_tensors_x;

        int globalBlockRow = sub_tensor_row * block_ROWS_packed;
        int globalBlockCol = sub_tensor_col * block_COLS_packed;

        //Special Values in BF16
        //Value         Exponent (E)        Mantissa (M)        Notes
        //Zero          00000000                0000000         Positive or negative (sign bit)
        //Inf           11111111                0000000         Positive or negative (sign bit)
        //NaN           11111111                non-zero            Any non-zero mantissa is NaN
        //Subnormals    00000000                non-zero            Denormals / subnormals
        //Normal            00000001–11111110   any                 Standard normalized values


        for (int r = 0; r < block_ROWS_packed; r += blockDim.y)
        {
            int row = globalBlockRow + r + threadIdx.y;

            for (int c = 0; c < block_COLS_packed; c += blockDim.x)
            {
                int col = globalBlockCol + c + threadIdx.x;

                // Load value from global memory
                uint32_t value = d_A_packed[row * N_packed + col];
                uint32_t bf16_value1 = value >> 16;
                uint32_t bf16_value2 = value & 0xFFFF;

                float fp32_value1 = bf16_to_fp32(bf16_value1);
                float fp32_value2 = bf16_to_fp32(bf16_value2);

                //if ( (row == 0) && (col == 172)) printf("fp32_value2 = %e\n", fp32_value2);

                float dequantized_value1 = 0.0f;
                float dequantized_value2 = 0.0f;

                if (fp32_value1 != 0)
                {
                    //S1_E8_M23
                    uint32_t bits = *reinterpret_cast<uint32_t*>(&fp32_value1);
                    uint32_t sign = (bits >> 31) & 0x1;
                    int32_t exponent =  (bits >> 23) & 0xFF;
                    uint32_t mantissa = (bits >> 16) & 0x7F;

                    if (exponent == 0) mantissa = 0; //resetting small values to 0... Actually can leave them alone too..
                    else if (exponent != 255)
                    {
                        if ( /*(0 <= mantissa) &&*/ (mantissa <= 32)) mantissa = 0;
                        else if ((33 <= mantissa) && (mantissa <= 95)) mantissa = 1 << 6;
                        else
                        {
                            mantissa = 0;
                            exponent++;
                            if (exponent == 255)
                            {
                                exponent = 254;
                                mantissa = 1 << 6;
                            }
                        }
                    }
                    bits = (sign << 31) | (exponent << 23) | (mantissa << 16);
                    dequantized_value1 = *reinterpret_cast<float*>(&bits);
                    //float here_error = fabsf(1.0 - dequantized_val/val);
                    //float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                    //local_error += here_error;
                }

                if (fp32_value2 != 0)
                {
                    //S1_E8_M23
                    uint32_t bits = *reinterpret_cast<uint32_t*>(&fp32_value2);
                    uint32_t sign = (bits >> 31) & 0x1;
                    int32_t exponent =  (bits >> 23) & 0xFF;
                    uint32_t mantissa = (bits >> 16) & 0x7F;

                    if (exponent == 0) mantissa = 0; //resetting small values to 0... Actually can leave them alone too..
                    else if (exponent != 255)
                    {
                        if ( /*(0 <= mantissa) &&*/ (mantissa <= 32)) mantissa = 0;
                        else if ((33 <= mantissa) && (mantissa <= 95)) mantissa = 1 << 6;
                        else
                        {
                            mantissa = 0;
                            exponent++;
                            if (exponent == 255)
                            {
                                exponent = 254;
                                mantissa = 1 << 6;
                            }
                        }
                    }
                    bits = (sign << 31) | (exponent << 23) | (mantissa << 16);
                    dequantized_value2 = *reinterpret_cast<float*>(&bits);
                    //float here_error = fabsf(1.0 - dequantized_val/val);
                    //float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                    //local_error += here_error;
                }

                uint32_t pruned_bf16_value1 = fp32_to_bf16(dequantized_value1);
                uint32_t pruned_bf16_value2 = fp32_to_bf16(dequantized_value2);
                uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;

                //if (sub_tensor_id == 0) printf("[row = %d :: col = %d] :: tid = %d :: val = %e :: scaling_factor = %e :: scaled_val = %e :: quantized_val = %e :: dequantized_val = %e :: here_error = %e\n", row, col, tid, val, scaling_factor, scaled_val, quantized_val, dequantized_val, here_error);

                // Write quantized value to global memory
                d_B_packed[row * N_packed + col] = merged_bf16;
            }
        }
    }
}

// Host Function to Launch the Kernel
void Phases1And2(const float* A, float *B, int M, int N, int block_ROWS, int block_COLS)
{
    ASSERT((M % block_ROWS) == 0);
    ASSERT((N % block_COLS) == 0);

    ASSERT((block_COLS % 2) == 0);

    int M_packed = M;
    int N_packed = N/2; //We are packing 2 bf16's in 1 fp32

    int block_ROWS_packed = block_ROWS;
    int block_COLS_packed = block_COLS/2;

    printf("M_packed = %d :: N_packed = %d\n", M_packed, N_packed);
    printf("block_ROWS_packed = %d :: block_COLS_packed = %d\n", block_ROWS_packed, block_COLS_packed);

    uint32_t *A_packed = (uint32_t *)malloc(M_packed * N_packed * sizeof(uint32_t));
    uint32_t *B_packed = (uint32_t *)malloc(M_packed * N_packed * sizeof(uint32_t));

    if (!A_packed || !B_packed) { fprintf(stderr, "Memory allocation failed!\n"); exit(EXIT_FAILURE); }

    {
        uint32_t *A_uint32_t = (uint32_t *)(A);
        for(int i = 0; i < M_packed * N_packed; i++)
        {
            uint32_t x = A_uint32_t[2*i + 0];
            uint32_t y = A_uint32_t[2*i + 1];

            uint16_t bf16_x = (x) >> 16;
            uint16_t bf16_y = (y) >> 16;

            ASSERT( (x & 0xFFFF) == 0);
            ASSERT( (y & 0xFFFF) == 0);

            uint32_t packed = ((uint32_t)bf16_x << 16) | bf16_y;
            A_packed[i] = packed;
            //if (A_packed[i] != 0) printf("[%d] --> %u\n", i, A_packed[i]);
        }
    }

    uint32_t *d_A_packed, *d_B_packed;

    int sub_tensors_x = N_packed / block_COLS_packed;
    int sub_tensors_y = M_packed / block_ROWS_packed;
    int total_sub_tensors = sub_tensors_x * sub_tensors_y;

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_packed, M_packed * N_packed * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_packed, M_packed * N_packed * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A_packed, A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

#if 0
#endif

#if 0
#endif

    int threads_x = 32;
    int threads_y = 8;

    ASSERT((block_ROWS_packed % threads_y) == 0);
    ASSERT((block_COLS_packed % threads_x) == 0);

    //std::cout << "Dynamically determined max blocks (considering GPU usage): " << adaptiveBlocks << std::endl;
    int numBlocks = total_sub_tensors;

    int threads_per_block = threads_x * threads_y;
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim(min(numBlocks, total_sub_tensors), 1);  // Ensure grid fits total sub-tensors

    ASSERT((threads_per_block % 32) == 0);

    //int sharedMemSize = (threads_per_block * sizeof(float)) + (threads_per_block * sizeof(int));
    int sharedMemSize = 0; //(warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));
    int number_of_sub_tensors_per_block = (total_sub_tensors % numBlocks) ? (total_sub_tensors/numBlocks) + 1 : (total_sub_tensors/numBlocks);

    // Debug prints for verification
    {
        printf("DEBUG: GPU Info\n");
        //printf("  Number of SMs: %d\n", numSMs);
        printf("  Total sub-tensors: %d\n", total_sub_tensors);
        //printf("  Blocks Per SM: 4\n");
        printf("DEBUG: Launch Configuration\n");
        printf("  numBlocks: %d\n", numBlocks);
        printf("  Num_Sub_Tensors Per Block: %d\n", number_of_sub_tensors_per_block);
        printf("  gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("  blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("  Shared memory size per block: %d bytes\n", sharedMemSize);
    }

    auto start_gpu = std::chrono::high_resolution_clock::now();
    int max_iterations = 100;
    for(int iter = 0; iter < max_iterations; iter++)
    {
        compute_E8M1_Representation<<<gridDim, blockDim, sharedMemSize>>>(
                d_A_packed, d_B_packed, M_packed, N_packed, block_ROWS_packed, block_COLS_packed, sub_tensors_x, sub_tensors_y);
    }
    cudaDeviceSynchronize();


    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> gpu_duration = end_gpu - start_gpu;
    double avg_gpu_time = gpu_duration.count()/max_iterations;

    PRINT_RED;
    printf("------------------------------------------------------------------------------------------------------\n");
    std::cout << ">>>>>>>>>>>>>> Average GPU computation completed!  Time (averaged over " << max_iterations << " iterations): "<< avg_gpu_time << " micro-seconds Per Iteration" << std::endl;
    {
        double effective_bandwidth = 0;
        effective_bandwidth = sizeof(uint32_t) * (M_packed * N_packed) * 2; //2-Reads + 1-Write
        effective_bandwidth /= avg_gpu_time;
        effective_bandwidth *= (1000 * 1000)/1024/1024.0/1024.0;
        printf(">>>>>>>>>>>>>> Effective B/W = %f GB/sec\n", effective_bandwidth);
    }
    printf("------------------------------------------------------------------------------------------------------\n");
    PRINT_RESET;

    CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_B_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    {

        for(int i = 0; i < M_packed * N_packed; i++)
        {
            uint32_t x = B_packed[i];

            uint32_t y = (x & 0xFFFF0000);
            B[2*i + 0] = *reinterpret_cast<float*>(&y);

            y = x << 16;
            B[2*i + 1] = *reinterpret_cast<float*>(&y);
        }
    }

    free(A_packed);
    free(B_packed);

    CHECK_CUDA_ERROR(cudaFree(d_A_packed));
    CHECK_CUDA_ERROR(cudaFree(d_B_packed));
}


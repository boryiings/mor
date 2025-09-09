

#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cooperative_groups.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <stdio.h>
#include <type_traits>
#include <vector>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
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
        int M, int N, 
        int block_ROWS, int block_COLS, 
        int number_of_sub_tensors_per_block, float *d_amax)
{
    //int threads_x = blockDim.x;
    //int threads_y = blockDim.y;
    /////int threads_per_block = threads_x * threads_y;
    //if (block_id == 0 && tid == 0) printf("number_of_sub_tensors_per_block = %d\n", number_of_sub_tensors_per_block);

    fp32 global_scaling_factor = 1.0;
    if (1)
    {
        fp32 amax = *d_amax;
        assert(amax > 0);
        uint32_t y = *reinterpret_cast<uint32_t*>(&amax);
        uint32_t exponent = (y >> 23);
        //uint32_t mantissa = y & 0x7FFFFF;
        exponent++;
        assert(exponent < 255); 
        uint32_t y2 = exponent << 23;
        y2 = y2 | (1 << 22);
        fp32 fp_max = *reinterpret_cast<fp32*>(&y2);
        global_scaling_factor = fp_max / amax;
        if (global_scaling_factor >= 2)
        {
            fp_max /= 2;
            global_scaling_factor /= 2;
        }

#if 0
        printf("amax = %e (Hex: 0x%x) :: fp_max = %e (Hex: 0x%x) :: global_scaling_factor = %e (Hex: 0x%x)\n", 
                amax, *reinterpret_cast<uint32_t*>(&amax), 
                fp_max, *reinterpret_cast<uint32_t*>(&fp_max),
                global_scaling_factor, *reinterpret_cast<uint32_t*>(&global_scaling_factor));
#endif
    }


    int start_index_x = (N / gridDim.x) * blockIdx.x;
    int   end_index_x = start_index_x + (N / gridDim.x);

    int start_index_y = (M / gridDim.y) * blockIdx.y;
    int   end_index_y = start_index_y + (M / gridDim.y);

    int N_packed = N/2;
    start_index_x /= 2;
      end_index_x /= 2;

    ///if (block_id == 0 && tid == 0) {}////printf("start_index_x = %d :: end_index_x = %d :: start_index_y = %d :: end_index_y = %d\n", start_index_x, end_index_x, start_index_y, end_index_y);
    for(int r = start_index_y; r < end_index_y; r += blockDim.y)
    {
        int row = r + threadIdx.y;
        for(int c = start_index_x; c < end_index_x; c += blockDim.x)
        {
            int col = c + threadIdx.x;
            //if ( (row >= M) || (col >= N)) { printf("block_id = %d :: tid = %d :: row = %d :: col = %d\n", block_id, tid, row, col); }

            uint32_t value = d_A_packed[row * N_packed + col];

            uint32_t bf16_value1 = value >> 16;
            uint32_t bf16_value2 = value & 0xFFFF;

            float fp32_value1 = bf16_to_fp32(bf16_value1);
            float fp32_value2 = bf16_to_fp32(bf16_value2);

            //Special Values in BF16
            //Value	        Exponent (E)	    Mantissa (M)	Notes
            //Zero	        00000000	        0000000	        Positive or negative (sign bit)
            //Inf	        11111111	        0000000	        Positive or negative (sign bit)
            //NaN	        11111111	        non-zero	    Any non-zero mantissa is NaN
            //Subnormals	00000000	        non-zero	    Denormals / subnormals
            //Normal	    00000001–11111110	any	            Standard normalized values

            float dequantized_value1 = 0.0f;
            float dequantized_value2 = 0.0f;

            if (fp32_value1 != 0)
            {
                fp32_value1 *= global_scaling_factor;
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
                dequantized_value1 /= global_scaling_factor;
                //float here_error = fabsf(1.0 - dequantized_val/val);
                //float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                //local_error += here_error;
            }

            if (fp32_value2 != 0)
            {
                fp32_value2 *= global_scaling_factor;
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
                dequantized_value2 /= global_scaling_factor;
                //float here_error = fabsf(1.0 - dequantized_val/val);
                //float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                //local_error += here_error;
            }

            uint32_t pruned_bf16_value1 = fp32_to_bf16(dequantized_value1);
            uint32_t pruned_bf16_value2 = fp32_to_bf16(dequantized_value2);
            uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;

            // Write quantized value to global memory
            d_B_packed[row * N_packed + col] = merged_bf16;
        }
    }
}


__global__ void compute_E8M1_Representation_Older(
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
        //Value	        Exponent (E)	    Mantissa (M)	Notes
        //Zero	        00000000	        0000000	        Positive or negative (sign bit)
        //Inf	        11111111	        0000000	        Positive or negative (sign bit)
        //NaN	        11111111	        non-zero	    Any non-zero mantissa is NaN
        //Subnormals	00000000	        non-zero	    Denormals / subnormals
        //Normal	    00000001–11111110	any	            Standard normalized values


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

// Reduction helper function (amax and non-zero counts)
__device__ void blockReduce(float& local_amax, int& local_non_zero, float* s_amax, int* s_non_zero, int tid) {
    unsigned int mask = __activemask();

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_amax = fmaxf(local_amax, __shfl_down_sync(mask, local_amax, offset));
        local_non_zero += __shfl_down_sync(mask, local_non_zero, offset);
    }

    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_amax[threadIdx.y * blockDim.x / 32 + threadIdx.x / 32] = local_amax;
        s_non_zero[threadIdx.y * blockDim.x / 32 + threadIdx.x / 32] = local_non_zero;
    }
    __syncthreads();

    // Shared memory reduction
    if (tid < (blockDim.x * blockDim.y / 32)) {
        local_amax = s_amax[tid];
        local_non_zero = s_non_zero[tid];
    } else {
        local_amax = -INFINITY;
        local_non_zero = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
            s_non_zero[tid] += s_non_zero[tid + stride];
        }
        __syncthreads();
    }
}


__global__ void compute_block_amax_and_non_zero(
    const uint32_t* d_A_packed, 
    int M, int N, int block_ROWS, int block_COLS, int number_of_sub_tensors_per_block, 
    float* d_block_amax, int* d_block_non_zero
    )
{
    int threads_x = blockDim.x;
    int threads_y = blockDim.y;
    int threads_per_block = threads_x * threads_y;
    //if (block_id == 0 && tid == 0) printf("number_of_sub_tensors_per_block = %d\n", number_of_sub_tensors_per_block);

    int start_index_x = (N / gridDim.x) * blockIdx.x;
    int   end_index_x = start_index_x + (N / gridDim.x);

    int start_index_y = (M / gridDim.y) * blockIdx.y;
    int   end_index_y = start_index_y + (M / gridDim.y);

    int N_packed = N/2;
    start_index_x /= 2;
      end_index_x /= 2;

    ///if (block_id == 0 && tid == 0) {}////printf("start_index_x = %d :: end_index_x = %d :: start_index_y = %d :: end_index_y = %d\n", start_index_x, end_index_x, start_index_y, end_index_y);
    float local_amax1 = 0.0;
    float local_amax2 = 0.0;

    int local_nnz1 = 0;
    int local_nnz2 = 0;

    for(int r = start_index_y; r < end_index_y; r += blockDim.y)
    {
        int row = r + threadIdx.y;
        for(int c = start_index_x; c < end_index_x; c += blockDim.x)
        {
            int col = c + threadIdx.x;
            //if ( (row >= M) || (col >= N)) { printf("block_id = %d :: tid = %d :: row = %d :: col = %d\n", block_id, tid, row, col); }

            uint32_t value = d_A_packed[row * N_packed + col];

            uint32_t bf16_value1 = value >> 16;
            uint32_t bf16_value2 = value & 0xFFFF;

            float fp32_value1 = bf16_to_fp32(bf16_value1);
            float fp32_value2 = bf16_to_fp32(bf16_value2);

            float abs_value1 = fabsf(fp32_value1);
            float abs_value2 = fabsf(fp32_value2);

            local_amax1 = fmaxf(local_amax1, abs_value1);
            local_amax2 = fmaxf(local_amax2, abs_value2);

            if (abs_value1 > 0) local_nnz1++;
            if (abs_value2 > 0) local_nnz2++;
        }
    }

    {
        //int sub_tensors_x = N / block_COLS;
        int sub_tensors_y = M / block_ROWS;
        assert(sub_tensors_y == gridDim.y);
    }

    //int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    {
        assert((threads_per_block % 32) == 0);
        int warps_per_block = threads_per_block/32;
        float local_amax = fmaxf(local_amax1, local_amax2);
        int local_nnz = local_nnz1 + local_nnz2;

        float* s_amax = sdata;
        int* s_nnz = (int *)((char *)sdata + (warps_per_block * sizeof(float)));

        blockReduce(local_amax, local_nnz, s_amax, s_nnz, tid);

        int sub_tensor_id = block_id;

        if (tid == 0)
        {
            d_block_amax[sub_tensor_id] = s_amax[0];
            d_block_non_zero[sub_tensor_id] = s_nnz[0];
        }
    }
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ int warp_reduce_sum(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__device__ void blockReduceAmaxNonZero(float local_amax, int local_non_zero, float* s_amax, int* s_non_zero, int flattened_tid) {
    // Step 1: Perform warp-level reduction within each warp
    local_amax = warp_reduce_max(local_amax);
    local_non_zero = warp_reduce_sum(local_non_zero);

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
    local_amax = warp_reduce_max(local_amax);
    local_non_zero = warp_reduce_sum(local_non_zero);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
    s_amax[0] = local_amax;
    s_non_zero[0] = local_non_zero;
    }
    __syncthreads();
}


__global__ void compute_global_amax_and_non_zero(
        float *block_amax_device, int *block_non_zero_device,
        float *global_amax_device, int *global_non_zero_device,
        int num_blocks)
{
    float global_amax = 0;
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

    if (tid == 0) 
    {
        global_amax_device[0] = global_amax;
        global_non_zero_device[0] = global_non_zero;
        //assert(global_non_zero > 0);
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
    float *d_block_amax, *d_global_amax;
    int *d_block_non_zero, *d_global_non_zero;

    int sub_tensors_x = N_packed / block_COLS_packed;
    int sub_tensors_y = M_packed / block_ROWS_packed;
    int total_sub_tensors = sub_tensors_x * sub_tensors_y;

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_packed, M_packed * N_packed * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_packed, M_packed * N_packed * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_amax, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_amax, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_non_zero,  total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_non_zero, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A_packed, A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int threads_x = 16;
    int threads_y = 16;

    ASSERT((block_ROWS_packed % threads_y) == 0);
    ASSERT((block_COLS_packed % threads_x) == 0);

    int numBlocks_x = sub_tensors_x;
    int numBlocks_y = sub_tensors_y;
    int numBlocks = numBlocks_x * numBlocks_y;

    int threads_per_block = threads_x * threads_y;
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim(numBlocks_x, numBlocks_y);

    ASSERT((threads_per_block % 32) == 0);

    int number_of_sub_tensors_per_block = 1;

    int sharedMemSize = 0;
    ASSERT((threads_per_block % 32) == 0);
    int warps_per_block = threads_per_block/32;
    sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    dim3 reductionBlockDim(1024);
    dim3 reductionGridDim(1);

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
        compute_block_amax_and_non_zero<<<gridDim, blockDim, sharedMemSize>>>(
                d_A_packed, M, N, block_ROWS, block_COLS, number_of_sub_tensors_per_block,
                d_block_amax, d_block_non_zero);

        compute_global_amax_and_non_zero<<<reductionGridDim, reductionBlockDim>>>(
                d_block_amax, d_block_non_zero, 
                d_global_amax, d_global_non_zero, total_sub_tensors);


        compute_E8M1_Representation<<<gridDim, blockDim, sharedMemSize>>>( 
                d_A_packed, d_B_packed, M, N, block_ROWS, block_COLS,
                number_of_sub_tensors_per_block, d_global_amax);
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

    float global_amax = 0;
    int global_non_zero = 0;

    CHECK_CUDA_ERROR(cudaMemcpy(&global_amax, d_global_amax, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_non_zero, d_global_non_zero, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_B_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("Global Amax = %e (Hex: 0x%x)\n", global_amax, *reinterpret_cast<uint32_t*>(&global_amax));
    printf("Global Non-Zero Count: %d\n", global_non_zero);

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

    CHECK_CUDA_ERROR(cudaFree(d_block_amax));
    CHECK_CUDA_ERROR(cudaFree(d_global_amax));

    CHECK_CUDA_ERROR(cudaFree(d_block_non_zero));
    CHECK_CUDA_ERROR(cudaFree(d_global_non_zero));

    CHECK_CUDA_ERROR(cudaFree(d_A_packed));
    CHECK_CUDA_ERROR(cudaFree(d_B_packed));
}

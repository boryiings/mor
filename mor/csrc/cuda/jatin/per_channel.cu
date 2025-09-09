
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

// Function to get the number of SMs on the device
int getNumberOfSMs() {
    int device;
    cudaGetDevice(&device);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    return numSMs;
}


// FP8-E4M3 conversion function

// FP8-E4M3 Constructor
__device__ uint8_t fp8_e4m3(float f)
{
    ///if (f == 448) printf("ASDF\n");
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 7; // E4 exponent bias is 7
    uint32_t mantissa = (bits >> 20) & 0x7;

    if (exponent <= 0)
    {
        //Dealing with sub_normal...
        fp32 f2 = (f < 0) ? -f : f;
        exponent = 0;

        if (f2 < (1/16.0 * 1/64.0)) mantissa = 0;
        else if (f2 < (3/16.0 * 1/64.0)) mantissa = 1;
        else if (f2 < (5/16.0 * 1/64.0)) mantissa = 2;
        else if (f2 < (7/16.0 * 1/64.0)) mantissa = 3;
        else if (f2 < (9/16.0 * 1/64.0)) mantissa = 4;
        else if (f2 < (11/16.0 * 1/64.0)) mantissa = 5;
        else if (f2 < (13/16.0 * 1/64.0)) mantissa = 6;
        else if (f2 < (15/16.0 * 1/64.0)) mantissa = 7;
        else
        {
            exponent = 1;
            mantissa = 0;
        }
    }
    else
    {
        uint32_t extra_mantissa = (bits >> 19) & 0x1;
        //assert(((extra_mantissa == 0) || (extra_mantissa == 1)));
        if (extra_mantissa == 1)
        {
            mantissa++;
            if (mantissa == 8)
            {
                exponent++;
                mantissa = 0;
            }
        }

        //if (exponent == 16) { printf("Error\n"); }

        if (exponent <= 0)  // Subnormal numbers
        {
            exponent = 0;
            mantissa = (bits >> 23) & 0x7;
        }
        else if (exponent > 15)
        { // Infinity or NaN
            exponent = 15;
            mantissa = 7;
            //printf("Found a NaN NaN NaN Nan\n");
            //ERROR_PRINT();
        }
        else if (exponent == 15)
        {
            if (mantissa == 7)
            {
                //printf("Found a NaN NaN NaN Nan\n");
                //ERROR_PRINT();
            }
        }
    }

    uint8_t value = (sign << 7) | (exponent << 3) | mantissa;
    return value;
}

__device__ uint8_t fp8_e4m3_NOT_USED(float f)
{
    const int exponent_bias = 7;
    //const float max_normal = 448.0f; // S.1111.110
    const float min_normal = 1.0f / 64.0f; // S.0001.000
    ////const float min_subnormal = 1.0f / 512.0f; // S.0000.001

    if (f == 0.0f) return 0; // Zero case

    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 0x1; // Extract sign bit
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + exponent_bias; // Compute biased exponent
    uint32_t mantissa = (bits >> 20) & 0x7; // Extract 3-bit mantissa

    // Subnormal Handling
    if (exponent <= 0) {
        float abs_f = fabsf(f);

        if (abs_f < (1.0f / 1024.0f)) return 0; // Map anything < 2^-10 to zero

        if (abs_f < (1.0f / 512.0f)) return (sign << 7) | 0x01; // Smallest FP8 subnormal

        // Map to subnormal range
        uint8_t subnormal_mantissa = (uint8_t)(abs_f / min_normal * 8.0f + 0.5f);
        return (sign << 7) | (subnormal_mantissa & 0x7); // Pack FP8 subnormal
    }

    // Normal Handling
    if (exponent > 15) { // Clamp to max value
        return (sign << 7) | 0x7F;
    }

    // Rounding Logic for Normal Numbers
    uint32_t extra_mantissa = (bits >> 19) & 0x1;
    if (extra_mantissa) {
        mantissa++;
        if (mantissa == 8) { // Carry overflow
            exponent++;
            mantissa = 0;
        }
    }

    // Clamp exponent if it overflows
    if (exponent > 15) exponent = 15;

    return (sign << 7) | (exponent << 3) | mantissa; // Pack FP8 bits
}

// Convert FP8-E4M3 to FP32
__device__ float fp8_e4m3_to_fp32(uint8_t value)
{
    const int exponent_bias = 7;
    uint32_t sign = (value >> 7) & 0x1;
    int32_t exponent = ((value >> 3) & 0xF) - exponent_bias + 127; // Convert back to FP32 exponent
    uint32_t mantissa = value & 0x7; // Extract 3-bit mantissa

    if (((value >> 3) & 0xF) == 0) { // Subnormal case
        float subnormal = (mantissa / 8.0f) * (1.0f / 64.0f);
        return sign ? -subnormal : subnormal;
    }

    // Normal FP32 reconstruction
    uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 20);
    return *reinterpret_cast<float*>(&bits);

}


__device__ float new_fp8_e4m3(float input_val)
{
    uint8_t quantized_val = fp8_e4m3(input_val);
    float dequantized_val = fp8_e4m3_to_fp32(quantized_val);
    return dequantized_val;
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

// Reduction helper function (amax and non-zero counts)
__device__ void blockReduce2(int& local_non_zero, int* s_non_zero, int tid) {
    unsigned int mask = __activemask();

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_non_zero += __shfl_down_sync(mask, local_non_zero, offset);
    }

    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_non_zero[threadIdx.y * blockDim.x / 32 + threadIdx.x / 32] = local_non_zero;
    }
    __syncthreads();

    // Shared memory reduction
    if (tid < (blockDim.x * blockDim.y / 32)) {
        local_non_zero = s_non_zero[tid];
    } else {
        local_non_zero = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_non_zero[tid] += s_non_zero[tid + stride];
        }
        __syncthreads();
    }
}

// Reduction helper function (amax and non-zero counts)
__device__ void blockReduce3(float& local_error, float* s_error, int tid) {
    unsigned int mask = __activemask();

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_error += __shfl_down_sync(mask, local_error, offset);
    }

    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_error[threadIdx.y * blockDim.x / 32 + threadIdx.x / 32] = local_error;
    }
    __syncthreads();

    // Shared memory reduction
    if (tid < (blockDim.x * blockDim.y / 32)) {
        local_error = s_error[tid];
    } else {
        local_error = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_error[tid] += s_error[tid + stride];
        }
        __syncthreads();
    }
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


// Reduction helper function (amax and non-zero counts)
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

__device__ void reduce_amax_nnz_per_group(
    float local_amax, int local_nnz,
    float& out_amax, int& out_nnz,
    int group_size)
{
    assert(group_size <= 32);

    const int tid = threadIdx.x;
    //const int group_id = tid / group_size;
    const int local_id = tid % group_size;

    // Perform reduction within the group using warp shuffle
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        float shfl_amax = __shfl_down_sync(0xffffffff, local_amax, offset, group_size);
        int   shfl_nnz  = __shfl_down_sync(0xffffffff, local_nnz, offset, group_size);

        if (local_id + offset < group_size) {
            local_amax = fmaxf(local_amax, shfl_amax);
            local_nnz  += shfl_nnz;
        }
    }

    // Output result in first thread of each group
    if (local_id == 0) {
        out_amax = local_amax;
        out_nnz  = local_nnz;
    }
}

__inline__ __device__ float warpReduceSum(float val) {
    // Perform warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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


__global__ void compute_global_e4m3_relative_error(
    float* block_error_device,
    float* global_error_device,
    int num_blocks) {

    float global_error = 0;

    int tid = threadIdx.x;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
    global_error += block_error_device[i];
    }

    ///int threads_per_block = blockDim.x;
    //int warps_per_block = threads_per_block / warpSize;
    extern __shared__ float sharedMem[];
    float* s_amax = sharedMem;

    blockReduceError(global_error, s_amax, tid);
    global_error = s_amax[0];

    if (tid == 0) {
    global_error_device[0] = global_error;
    }
}


__global__ void compute_maybe_use_bf16(
    const uint32_t* d_A_packed, uint32_t* d_B_packed,
    int M, int N, int block_ROWS, int block_COLS, int number_of_sub_tensors_per_block,
    float* global_error_device, int *global_non_zero_device,
    int* d_quant_type_used
    )
{
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float average_relative_error = global_error_device[0]/global_non_zero_device[0];
    if (average_relative_error > 0.045)
    {
        if (block_id == 0 && tid == 0) *d_quant_type_used = 2;
        int start_index_x = (N / gridDim.x) * blockIdx.x;
        int   end_index_x = start_index_x + (N / gridDim.x);

        int start_index_y = (M / gridDim.y) * blockIdx.y;
        int   end_index_y = start_index_y + (M / gridDim.y);

        int N_packed = N/2;
        start_index_x /= 2;
          end_index_x /= 2;


        for(int r = start_index_y; r < end_index_y; r += blockDim.y)
        {
            int row = r + threadIdx.y;
            for(int c = start_index_x; c < end_index_x; c += blockDim.x)
            {
                int col = c + threadIdx.x;
                d_B_packed[row * N_packed + col] = d_A_packed[row * N_packed + col];
            }
        }
    }
    else
    {
        if (block_id == 0 && tid == 0) *d_quant_type_used = 1;
    }
}



__global__ void compute_block_e4m3_relative_error(
    const uint32_t* d_A_packed, uint32_t* d_B_packed,
    int M, int N, int block_ROWS, int block_COLS, int number_of_sub_tensors_per_block, 
    float* d_block_amax, float* d_global_amax, int* d_block_non_zero, int* d_global_non_zero,
    float *d_block_error
    )
{

    int start_index_x = (N / gridDim.x) * blockIdx.x;
    int   end_index_x = start_index_x + (N / gridDim.x);

    int start_index_y = (M / gridDim.y) * blockIdx.y;
    int   end_index_y = start_index_y + (M / gridDim.y);

    int N_packed = N/2;
    start_index_x /= 2;
      end_index_x /= 2;

    //Step 1: Let's first compute the scaling factor for all the
    //blocks this block (of threads) is going to touch...
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int starting_sub_tensor_id = block_id * number_of_sub_tensors_per_block;

    uint32_t matrix_sf_mantissa = 0;
    {
        float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
        float global_amax = d_global_amax[0];  // Use reduced amax from shared memory
        float ratio = max_fp8_e4m3 / global_amax;
        matrix_sf_mantissa = *reinterpret_cast<uint32_t*>(&ratio) & 0x007FFFFF;

        //int tid = threadIdx.y * blockDim.x + threadIdx.x;
        //if (block_id == 0 && tid == 0) printf("matrix_sf_mantissa = %d\n", matrix_sf_mantissa);
        //if ((block_id == 0) && (tid == 0)) printf("global_amax = %e (0x%08X)\n", global_amax, *(reinterpret_cast<uint32_t*>(&global_amax)));
    }

    float scaling_factor1 = 1.0;
    float scaling_factor2 = 1.0;
    if (number_of_sub_tensors_per_block == 1)
    {
        float max_fp8_e4m3 = 448.0f;
        float block_sf = max_fp8_e4m3 / d_block_amax[starting_sub_tensor_id];
        uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
        uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
        float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
        scaling_factor1 = (adjusted_scaling_factor <= block_sf) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;
        scaling_factor2 = scaling_factor1;
    }
    else
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        if (block_COLS == 1)
        {
            float max_fp8_e4m3 = 448.0f;
            float block_sf = max_fp8_e4m3 / d_block_amax[starting_sub_tensor_id + 2 * tx + 0];
            uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
            uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
            float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
            scaling_factor1 = (adjusted_scaling_factor <= block_sf) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;

            block_sf = max_fp8_e4m3 / d_block_amax[starting_sub_tensor_id + 2 * tx + 1];
            block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
            adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
            adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
            scaling_factor2 = (adjusted_scaling_factor <= block_sf) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;
        }
        else
        {
            //int threads_x = blockDim.x;
            int threads_y = blockDim.y;
            //int threads_per_block = threads_x * threads_y;
            assert(block_ROWS == 1); assert(threads_y == 1); assert(ty == 0);
            int group_size = block_COLS/2;
            int group_id = tx / group_size;

            float max_fp8_e4m3 = 448.0f;
            float block_sf = max_fp8_e4m3 / d_block_amax[starting_sub_tensor_id + group_id];
            uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
            uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
            float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
            scaling_factor1 = (adjusted_scaling_factor <= block_sf) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;
            scaling_factor2 = scaling_factor1;
        }
    }

    fp32 local_error1 = 0.0;
    fp32 local_error2 = 0.0;

    for(int r = start_index_y; r < end_index_y; r += blockDim.y)
    {
        int row = r + threadIdx.y;
        for(int c = start_index_x; c < end_index_x; c += blockDim.x)
        {
            int col = c + threadIdx.x;

            uint32_t value = d_A_packed[row * N_packed + col];

            uint32_t bf16_value1 = value >> 16;
            uint32_t bf16_value2 = value & 0xFFFF;

            float fp32_value1 = bf16_to_fp32(bf16_value1);
            float fp32_value2 = bf16_to_fp32(bf16_value2);

            float dequantized_value1 = 0.0f;
            float dequantized_value2 = 0.0f;

            if (fp32_value1 != 0)
            {
                float scaled_value1 = fp32_value1 * scaling_factor1;
                fp32 quantized_value1 = new_fp8_e4m3(scaled_value1);
                dequantized_value1 = quantized_value1 / scaling_factor1;
                float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                local_error1 += here_error;
            }

            if (fp32_value2 != 0)
            {
                float scaled_value2 = fp32_value2 * scaling_factor2;
                fp32 quantized_value2 = new_fp8_e4m3(scaled_value2);
                dequantized_value2 = quantized_value2 / scaling_factor2;
                float here_error = fabsf((dequantized_value2 - fp32_value2)/fp32_value2);
                local_error2 += here_error;
            }

            uint32_t pruned_bf16_value1 = fp32_to_bf16(dequantized_value1);
            uint32_t pruned_bf16_value2 = fp32_to_bf16(dequantized_value2);
            uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;
            d_B_packed[row * N_packed + col] = merged_bf16;
        }
    }

    extern __shared__ float sdata[];

    //if (number_of_sub_tensors_per_block == 1)
    if (1)
    {
        int starting_sub_tensor_id = block_id * number_of_sub_tensors_per_block;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        float local_error = local_error1 + local_error2;
        blockReduce3(local_error, sdata, tid);
        if (tid == 0) d_block_error[starting_sub_tensor_id + 0] = sdata[0];
        else if (tid < number_of_sub_tensors_per_block) d_block_error[starting_sub_tensor_id + tid] = 0.0;
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

    if (number_of_sub_tensors_per_block == 1)
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
    else
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        {
            float *smem1 = sdata;
            float *smem2 = smem1 + threads_x * threads_y;
            int idx = ty * threads_x + tx;

            smem1[idx] = local_amax1;
            smem2[idx] = local_amax2;
            __syncthreads();

            for (int stride = threads_y / 2; stride > 0; stride /= 2)
            {
                if (ty < stride)
                {
                    int upper = ty * threads_x + tx;
                    int lower = (ty + stride) * threads_x + tx;
                    smem1[upper] = fmaxf(smem1[upper], smem1[lower]);
                    smem2[upper] = fmaxf(smem2[upper], smem2[lower]);
                }
                __syncthreads();
            }

            if (ty == 0)
            {
                local_amax1 = smem1[tx];
                local_amax2 = smem2[tx];
            }
            __syncthreads();
        }
        {
            int *smem1 = (int *)(sdata);
            int *smem2 = smem1 + threads_x * threads_y;
            int idx = ty * threads_x + tx;

            smem1[idx] = local_nnz1;
            smem2[idx] = local_nnz2;
            __syncthreads();

            for (int stride = threads_y / 2; stride > 0; stride /= 2)
            {
                if (ty < stride)
                {
                    int upper = ty * threads_x + tx;
                    int lower = (ty + stride) * threads_x + tx;
                    smem1[upper] = smem1[upper] + smem1[lower];
                    smem2[upper] = smem2[upper] + smem2[lower];
                }
                __syncthreads();
            }

            if (ty == 0)
            {
                local_nnz1 = smem1[tx];
                local_nnz2 = smem2[tx];
            }
            __syncthreads();
        }

        int starting_sub_tensor_id = block_id * number_of_sub_tensors_per_block;
        if (block_COLS == 1)
        {
            if (ty == 0)
            {
                d_block_amax[starting_sub_tensor_id + 2 * tx + 0] = local_amax1;
                d_block_amax[starting_sub_tensor_id + 2 * tx + 1] = local_amax2;

                d_block_non_zero[starting_sub_tensor_id + 2 * tx + 0] = local_nnz1;
                d_block_non_zero[starting_sub_tensor_id + 2 * tx + 1] = local_nnz2;
            }
        }
        else //We in threadx.x each block_COLS/2 belongs to 1 block and we need to reduce across that...
        {
            assert(block_ROWS == 1); assert(threads_y == 1); assert(ty == 0);

            float local_amax = fmaxf(local_amax1, local_amax2);
            int local_nnz = local_nnz1 + local_nnz2;

            float final_amax = 0.0f;
            int final_nnz = 0;
            int group_size = block_COLS/2;

            reduce_amax_nnz_per_group(local_amax, local_nnz, final_amax, final_nnz, group_size);
            if ((tx % group_size) == 0)
            {
                int group_id = tx / group_size;
                d_block_amax[starting_sub_tensor_id + group_id] = final_amax;
                d_block_non_zero[starting_sub_tensor_id + group_id] = final_nnz;
            }
        }
    }
}


// Host Function to Launch the Kernel
void Phases1And2(const float* A, float *B, int M, int N, int block_ROWS, int block_COLS)
{
    printf("In (%s) :: M = %d :: N = %d :: block_ROWS = %d :: block_COLS = %d\n", __FILE__, M, N, block_ROWS, block_COLS);
    ASSERT((M % block_ROWS) == 0);
    ASSERT((N % block_COLS) == 0);
    ASSERT((block_ROWS >= 16) || (block_COLS >= 16));

    ///ASSERT((block_COLS % 2) == 0);

    int M_packed = M;
    int N_packed = N/2; //We are packing 2 bf16's in 1 fp32

    int block_ROWS_packed = block_ROWS;
    int block_COLS_packed = block_COLS/2;


    ASSERT( (block_ROWS == 1) || (block_COLS == 1));

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
    float *d_block_error, *d_global_error;
    int *d_block_non_zero; //, *d_block_flush_to_zero;
    int *d_global_non_zero; //, *d_global_flush_to_zero;
    int *d_quant_type_used;

    int sub_tensors_x = N / block_COLS;
    int sub_tensors_y = M / block_ROWS;
    int total_sub_tensors = sub_tensors_x * sub_tensors_y;

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_packed, M_packed * N_packed * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_packed, M_packed * N_packed * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_amax, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_amax, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_error, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_error, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_non_zero,  total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_non_zero, sizeof(int)));

    //CHECK_CUDA_ERROR(cudaMalloc(&d_block_flush_to_zero, total_sub_tensors * sizeof(int)));
    //CHECK_CUDA_ERROR(cudaMalloc(&d_global_flush_to_zero, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_quant_type_used, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A_packed, A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_A_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

#if 0
    {
        for(int i = 0; i < (M_packed * N_packed); i++) ASSERT(A_packed[i] == B_packed[i]);
        printf("A_packed[65] = %u\n", A_packed[65]);
        printf("B_packed[65] = %u\n", B_packed[65]);
    }
#endif

    ASSERT((block_ROWS * block_COLS) >= 16);

    int threads_x = 32;
    int threads_y = 16;
    if (block_ROWS == 1) { threads_x = 64;  threads_y = 1;}
    if (block_COLS == 1) { threads_x = 32;  threads_y = 16;} //We //assume we will have atleast 16 values in each block in Y axis

    //if (block_ROWS == 1) { threads_x = 128; }

    int threads_per_block = threads_x * threads_y;
    int number_of_sub_tensors_per_block = 1;
    if (block_COLS == 1) number_of_sub_tensors_per_block = (2*threads_x);
    else if (block_COLS < (2*threads_x)) number_of_sub_tensors_per_block = (2*threads_x)/block_COLS;

    if (block_ROWS == 1) 
    {
        //threads_x = 8;
        if (block_COLS < (2*threads_x)) { ASSERT(((2*threads_x) % block_COLS) == 0); }
        else { ASSERT(block_COLS % (2*threads_x) == 0); }
        if (number_of_sub_tensors_per_block > 1) ASSERT((block_COLS/2) <= 32); //we want it to fit within a warp if more than //1 block in X direction...
    }


    int numBlocks_x = sub_tensors_x / number_of_sub_tensors_per_block;
    int numBlocks_y = sub_tensors_y;
    int numBlocks = numBlocks_x * numBlocks_y;
    ASSERT(numBlocks_x >= 1); ASSERT(numBlocks_y >= 1);

    //if (block_COLS == 1) { ASSERT(block_ROWS >= 64); ASSERT((M % threads_y) == 0); }

    dim3 blockDim(threads_x, threads_y);  
    dim3 gridDim(numBlocks_x, numBlocks_y);

    //int threads_per_block = threads_x * threads_y;
    //ASSERT((threads_per_block % 32) == 0);
    //////int warps_per_block = threads_per_block/32;

    int sharedMemSize = 0;
    if (number_of_sub_tensors_per_block == 1) 
    {
        ASSERT((threads_per_block % 32) == 0);
        int warps_per_block = threads_per_block/32;
        sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));
    }
    else sharedMemSize = 2 * threads_x * threads_y * sizeof(float); 

    // Debug prints for verification
    {
        int numSMs = getNumberOfSMs();
        printf("DEBUG: GPU Info\n");
        printf("  Number of SMs: %d\n", numSMs);
        printf("  Total sub-tensors: %d\n", total_sub_tensors);
        printf("  Blocks Per SM: 4\n");
        printf("DEBUG: Launch Configuration\n");
        printf("  numBlocks: %d\n", numBlocks);
        printf("  Num_Sub_Tensors Per Block: %d\n", number_of_sub_tensors_per_block);
        printf("  gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("  blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("  Shared memory size per block: %d bytes\n", sharedMemSize);
    }

    dim3 reductionBlockDim(1024);
    dim3 reductionGridDim(1);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    int max_iterations = 1000;
    for(int iter = 0; iter < max_iterations; iter++)
    {
        compute_block_amax_and_non_zero<<<gridDim, blockDim, sharedMemSize>>>(
                d_A_packed, M, N, block_ROWS, block_COLS, number_of_sub_tensors_per_block,
                d_block_amax, d_block_non_zero);

        compute_global_amax_and_non_zero<<<reductionGridDim, reductionBlockDim>>>(
                d_block_amax, d_block_non_zero, 
                d_global_amax, d_global_non_zero, total_sub_tensors);

        compute_block_e4m3_relative_error<<<gridDim, blockDim, sharedMemSize>>>(
                d_A_packed, d_B_packed, M, N, block_ROWS, block_COLS, number_of_sub_tensors_per_block,
                d_block_amax, d_global_amax, d_block_non_zero, d_global_non_zero, 
                d_block_error);

        compute_global_e4m3_relative_error<<<reductionGridDim, reductionBlockDim>>>(
                d_block_error, d_global_error, total_sub_tensors);

        compute_maybe_use_bf16<<<gridDim, blockDim>>>(
                d_A_packed, d_B_packed, M, N, block_ROWS, block_COLS, number_of_sub_tensors_per_block,
                d_global_error, d_global_non_zero, d_quant_type_used);
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
        effective_bandwidth = sizeof(uint32_t) * (M_packed * N_packed) * (2 + 1); //2-Reads + 1-Write
        effective_bandwidth /= avg_gpu_time;
        effective_bandwidth *= (1000 * 1000)/1024/1024.0/1024.0;
        printf(">>>>>>>>>>>>>> Effective B/W = %f GB/sec\n", effective_bandwidth);
    }
    printf("------------------------------------------------------------------------------------------------------\n");
    PRINT_RESET;

    float global_amax = 0;
    float global_error = 0;
    int global_non_zero = 0;
    //int global_flush_to_zero = 0;
    int quant_type_used = 0;

    CHECK_CUDA_ERROR(cudaMemcpy(&global_non_zero, d_global_non_zero, sizeof(int), cudaMemcpyDeviceToHost));
    //CHECK_CUDA_ERROR(cudaMemcpy(&global_flush_to_zero, d_global_flush_to_zero, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_amax, d_global_amax, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_error, d_global_error, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&quant_type_used, d_quant_type_used, sizeof(int), cudaMemcpyDeviceToHost));

    //float global_error = 0;
    //CHECK_CUDA_ERROR(cudaMemcpy(&global_error, d_global_error, sizeof(float), cudaMemcpyDeviceToHost));

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


    printf("Global Non-Zero Count: %d\n", global_non_zero);
    //printf("Global Flush-To-Zero Count: %d\n", global_flush_to_zero);
    printf("Global Amax = %e (Hex: 0x%x)\n", global_amax, *reinterpret_cast<uint32_t*>(&global_amax));
    printf("Global Error = %e (Hex: 0x%x)\n", global_error, *reinterpret_cast<uint32_t*>(&global_error));
    //printf("Global Error = %e\n", global_error);
    float average_relative_error = global_error/global_non_zero;
    printf("Average Error = %e (Hex: 0x%x) :: quant_type_used = %d\n", average_relative_error, *reinterpret_cast<uint32_t*>(&average_relative_error), quant_type_used);

    free(A_packed);
    free(B_packed);

    CHECK_CUDA_ERROR(cudaFree(d_A_packed));
    CHECK_CUDA_ERROR(cudaFree(d_B_packed));

    CHECK_CUDA_ERROR(cudaFree(d_block_non_zero));
    //CHECK_CUDA_ERROR(cudaFree(d_block_flush_to_zero));
    CHECK_CUDA_ERROR(cudaFree(d_block_amax));
    CHECK_CUDA_ERROR(cudaFree(d_block_error));

    CHECK_CUDA_ERROR(cudaFree(d_global_non_zero));
    //CHECK_CUDA_ERROR(cudaFree(d_global_flush_to_zero));
    CHECK_CUDA_ERROR(cudaFree(d_global_amax));
    CHECK_CUDA_ERROR(cudaFree(d_global_error));
}

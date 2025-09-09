

#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cooperative_groups.h>

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





// Combined kernel for Phases 1 and 2
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
        )
{
    int threads_per_block = blockDim.x * blockDim.y;
    int warps_per_block = threads_per_block/32;

    /////printf("threads_per_block = %d :: warps_per_block = %d\n", threads_per_block, warps_per_block);

    int total_sub_tensors = sub_tensors_x * sub_tensors_y;
    extern __shared__ float sdata[];

    float* s_amax = sdata;
    int* s_non_zero = (int*)((char*)sdata + (warps_per_block * sizeof(float)));

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

#if 1
    //if (tid == 0) printf("block_id = %04d :: start_sub_tensor_id = %d :: end_sub_tensor_id = %d\n", block_id, start_sub_tensor_id, end_sub_tensor_id);
#endif

    ////if ( (block_id == 0) && (tid == 0)) *d_global_error = 0; //Someone //needs //to //initiaze //this //variable...

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 1: Compute local block maxima and non-zero counts
    //////////////////////////////////////////////////////////////////////////////////
    for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id) 
    {
        int sub_tensor_row = sub_tensor_id / sub_tensors_x;
        int sub_tensor_col = sub_tensor_id % sub_tensors_x;

        int globalBlockRow = sub_tensor_row * block_ROWS_packed;
        int globalBlockCol = sub_tensor_col * block_COLS_packed;

        float local_amax = -INFINITY;
        int local_non_zero = 0;

        for (int r = 0; r < block_ROWS_packed; r += blockDim.y)
        {
            int row = globalBlockRow + r + threadIdx.y;

            for (int c = 0; c < block_COLS_packed; c += blockDim.x)
            {
                int col = globalBlockCol + c + threadIdx.x;

                uint32_t value = d_A_packed[row * N_packed + col];
                //if ( (row == 0) && (col == 65)) printf("ZZZZ %u\n", value);
                //if ( value != 0)  printf("ZZZZ %u\n", value);

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
        blockReduce(local_amax, local_non_zero, s_amax, s_non_zero, tid);

        // Write results to global memory
        if (tid == 0) 
        {
            d_block_amax[sub_tensor_id] = s_amax[0];
            d_block_non_zero[sub_tensor_id] = s_non_zero[0];
        }
    }

    grid.sync();  // Ensure all blocks have written their block-level maxima and non-zero counts

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 2: Compute global maxima and non-zero counts
    //////////////////////////////////////////////////////////////////////////////////

    ////if ( (block_id == 0) && (tid == 0)) assert(*d_global_error == 0); //Verifying if it has been zet :)

    float global_amax = -INFINITY;
    int global_non_zero = 0;

    for (int i = tid; i < total_sub_tensors; i += blockDim.x * blockDim.y) 
    {
        global_amax = fmaxf(global_amax, d_block_amax[i]);
        global_non_zero += d_block_non_zero[i];
    }

    blockReduce(global_amax, global_non_zero, s_amax, s_non_zero, tid);

    global_non_zero = s_non_zero[0];

    // Write global results to memory (only one thread writes)
    if (block_id == 0 && tid == 0)
    {
        *d_global_amax = s_amax[0];
        *d_global_non_zero = s_non_zero[0];
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Phase 3: Quantization...
    //////////////////////////////////////////////////////////////////////////////////

#if 1
    uint32_t matrix_sf_mantissa = 0;
    {
        float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
        float global_amax = s_amax[0];  // Use reduced amax from shared memory
        float ratio = max_fp8_e4m3 / global_amax;
        matrix_sf_mantissa = *reinterpret_cast<uint32_t*>(&ratio) & 0x007FFFFF;
        //if (tid == 0) printf("matrix_sf_mantissa = %d\n", matrix_sf_mantissa);
    }


    // Iterate through the assigned sub-tensors for this block
    for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id)
    {
        int sub_tensor_row = sub_tensor_id / sub_tensors_x;
        int sub_tensor_col = sub_tensor_id % sub_tensors_x;

        int globalBlockRow = sub_tensor_row * block_ROWS_packed;
        int globalBlockCol = sub_tensor_col * block_COLS_packed;

        // Compute scaling factor for this sub-tensor
        float scaling_factor = 1.0; //initialize it to something...
        //float min_e4m3_representable_threshold = d_block_amax[sub_tensor_id];

        {
            float max_fp8_e4m3 = 448.0f;  // Max value in FP8-E4M3
            float block_sf = max_fp8_e4m3 / d_block_amax[sub_tensor_id];
            uint32_t block_sf_bits = *reinterpret_cast<uint32_t*>(&block_sf);
            uint32_t adjusted_bits = (block_sf_bits & 0xFF800000) | matrix_sf_mantissa;
            float adjusted_scaling_factor = *reinterpret_cast<float*>(&adjusted_bits);
            scaling_factor = (adjusted_scaling_factor <= block_sf) ? adjusted_scaling_factor : adjusted_scaling_factor / 2.0f;

            {
                uint32_t bits = *reinterpret_cast<uint32_t*>(&scaling_factor);
                uint32_t res_mantissa_bits = bits & 0x007FFFFF;
                assert(res_mantissa_bits == matrix_sf_mantissa);
            }

            //fp32 dynamic_range_of_e4m3_range_for_subtensor = 448.0 * 64; 
            //dynamic_range_of_e4m3_range_for_subtensor *= (scaling_factor/block_sf);
            //min_e4m3_representable_threshold = d_block_amax[sub_tensor_id]/dynamic_range_of_e4m3_range_for_subtensor;
        }

        //if (sub_tensor_id == 0) if (tid == 1) printf("sub_tensor_id = %d :: scaling_factor = %e\n", sub_tensor_id, scaling_factor);
        // Quantize values in the sub-tensor

        fp32 local_error = 0.0;
        //int local_flush_to_zero = 0;

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

                float dequantized_value1 = 0.0f;
                float dequantized_value2 = 0.0f;

                if (fp32_value1 != 0)
                {
                    // Quantize using the scaling factor
                    float scaled_value1 = fp32_value1 * scaling_factor;
                    fp32 quantized_value1 = new_fp8_e4m3(scaled_value1);
                    //if (quantized_val == 0) local_flush_to_zero++;
                    //if (abs_val < min_e4m3_representable_threshold) local_flush_to_zero++;

                    // Dequantize and compute relative error
                    dequantized_value1 = quantized_value1 / scaling_factor;
                    //float here_error = fabsf(1.0 - dequantized_val/val);
                    float here_error = fabsf((dequantized_value1 - fp32_value1)/fp32_value1);
                    local_error += here_error;
                }

                if (fp32_value2 != 0)
                {
                    float scaled_value2 = fp32_value2 * scaling_factor;
                    fp32 quantized_value2 = new_fp8_e4m3(scaled_value2);
                    dequantized_value2 = quantized_value2 / scaling_factor;
                    float here_error = fabsf((dequantized_value2 - fp32_value2)/fp32_value2);
                    local_error += here_error;
                }

                uint32_t pruned_bf16_value1 = fp32_to_bf16(dequantized_value1);
                uint32_t pruned_bf16_value2 = fp32_to_bf16(dequantized_value2);
                uint32_t merged_bf16 = pruned_bf16_value1 << 16 | pruned_bf16_value2;

                //if (sub_tensor_id == 0) printf("[row = %d :: col = %d] :: tid = %d :: val = %e :: scaling_factor = %e :: scaled_val = %e :: quantized_val = %e :: dequantized_val = %e :: here_error = %e\n", row, col, tid, val, scaling_factor, scaled_val, quantized_val, dequantized_val, here_error);

                // Write quantized value to global memory
                d_B_packed[row * N_packed + col] = merged_bf16;
            }
        }

        // Reduce within the block
        //blockReduce2(local_flush_to_zero, s_non_zero, tid);
        blockReduce3(local_error, s_amax, tid);

        // Write results to global memory
        if (tid == 0) 
        {
            //d_block_flush_to_zero[sub_tensor_id] = s_non_zero[0];
            d_block_error[sub_tensor_id] = s_amax[0];
        }
    }

    grid.sync();  // Ensure all blocks have written their block-level maxima and non-zero counts

    //int global_flush_to_zero = 0;
    float global_error = 0;

    for (int i = tid; i < total_sub_tensors; i += blockDim.x * blockDim.y) 
    {
        //global_flush_to_zero += d_block_flush_to_zero[i];
        global_error += d_block_error[i];
    }

    //blockReduce2(global_flush_to_zero, s_non_zero, tid);
    blockReduce3(global_error, s_amax, tid);

    // Write global results to memory (only one thread writes)
    if (block_id == 0 && tid == 0)
    {
        //*d_global_flush_to_zero = s_non_zero[0];
        *d_global_error = s_amax[0];
        *d_quant_type_used = 1;
    }

    global_error = s_amax[0];
    float average_relative_error = global_error/global_non_zero;

    float threshold_for_relative_error = 0.045;
    if (average_relative_error > threshold_for_relative_error)
    {
        if (block_id == 0 && tid == 0) *d_quant_type_used = 2;
        //We are going to store in BF16, and basically copy from A to B :)
        for (int sub_tensor_id = start_sub_tensor_id; sub_tensor_id < end_sub_tensor_id; ++sub_tensor_id) 
        {
            int sub_tensor_row = sub_tensor_id / sub_tensors_x;
            int sub_tensor_col = sub_tensor_id % sub_tensors_x;

            int globalBlockRow = sub_tensor_row * block_ROWS_packed;
            int globalBlockCol = sub_tensor_col * block_COLS_packed;

            for (int r = 0; r < block_ROWS_packed; r += blockDim.y)
            {
                int row = globalBlockRow + r + threadIdx.y;

                for (int c = 0; c < block_COLS_packed; c += blockDim.x)
                {
                    int col = globalBlockCol + c + threadIdx.x;
                    d_B_packed[row * N_packed + col] = d_A_packed[row * N_packed + col];
                }
            }
        }
    }
    //float fraction_of_flush_to_zero = (1.0 * global_flush_to_zero)/global_non_zero;
    //if ((block_id == 0) && (tid == 0)) printf("fraction_of_flush_to_zero = %e\n", fraction_of_flush_to_zero);
#endif
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
    float *d_block_error, *d_global_error;
    int *d_block_non_zero, *d_block_flush_to_zero;
    int *d_global_non_zero, *d_global_flush_to_zero;
    int *d_quant_type_used;

    int sub_tensors_x = N_packed / block_COLS_packed;
    int sub_tensors_y = M_packed / block_ROWS_packed;
    int total_sub_tensors = sub_tensors_x * sub_tensors_y;

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_packed, M_packed * N_packed * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_packed, M_packed * N_packed * sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_amax, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_amax, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_error, total_sub_tensors * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_error, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_non_zero,  total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_non_zero, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_block_flush_to_zero, total_sub_tensors * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_flush_to_zero, sizeof(int)));

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

    int numSMs = getNumberOfSMs();
    int numBlocks = numSMs * 2;  // 4 blocks per SM

    {
        if (total_sub_tensors % numBlocks) numBlocks = pow(2, floor(log2(numBlocks)) + 1);
        ASSERT(total_sub_tensors % numBlocks == 0);
    }

    int threads_x = 32;
    int threads_y = 8;

    int threads_per_block = threads_x * threads_y;
    dim3 blockDim(threads_x, threads_y);  // 256 threads per block
    dim3 gridDim(min(numBlocks, total_sub_tensors), 1);  // Ensure grid fits total sub-tensors

    ASSERT((threads_per_block % 32) == 0);
    int warps_per_block = threads_per_block/32;

    //int sharedMemSize = (threads_per_block * sizeof(float)) + (threads_per_block * sizeof(int));
    int sharedMemSize = (warps_per_block * sizeof(float)) + (warps_per_block * sizeof(int));

    // Debug prints for verification
    {
        printf("DEBUG: GPU Info\n");
        printf("  Number of SMs: %d\n", numSMs);
        printf("  Total sub-tensors: %d\n", total_sub_tensors);
        printf("  Blocks Per SM: 4\n");
        printf("DEBUG: Launch Configuration\n");
        printf("  numBlocks: %d\n", numBlocks);
        printf("  Num_Sub_Tensors Per Block: %d\n", total_sub_tensors/numBlocks);
        printf("  gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("  blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("  Shared memory size per block: %d bytes\n", sharedMemSize);
    }

    void* kernelArgs[] = {
        (void*)&d_A_packed, (void*)&d_B_packed,
        (void*)&M_packed, (void*)&N_packed, 
        (void*)&block_ROWS_packed, (void*)&block_COLS_packed, 
        (void*)&sub_tensors_x, (void*)&sub_tensors_y,
        (void*)&d_block_non_zero, (void*)&d_global_non_zero,
        (void*)&d_block_flush_to_zero,  (void *)&d_global_flush_to_zero, 
        (void*)&d_block_amax, (void*)&d_global_amax,
        (void*)&d_block_error, (void*)&d_global_error,
        (void*)&d_quant_type_used
    };

    auto start_gpu = std::chrono::high_resolution_clock::now();
    int max_iterations = 1000;

    for(int iter = 0; iter < max_iterations; iter++)
    {
        // Launch the kernel using cooperative kernel launch
        CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel( (void*)combinedKernelPhases1And2, gridDim, blockDim, kernelArgs, sharedMemSize));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

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
    int global_flush_to_zero = 0;
    int quant_type_used = 0;

    CHECK_CUDA_ERROR(cudaMemcpy(&global_non_zero, d_global_non_zero, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_flush_to_zero, d_global_flush_to_zero, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_amax, d_global_amax, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&global_error, d_global_error, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&quant_type_used, d_quant_type_used, sizeof(int), cudaMemcpyDeviceToHost));

    //float global_error = 0;
    //CHECK_CUDA_ERROR(cudaMemcpy(&global_error, d_global_error, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaMemcpy(B_packed, d_B_packed, M_packed * N_packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("Global Non-Zero Count: %d\n", global_non_zero);
    printf("Global Flush-To-Zero Count: %d\n", global_flush_to_zero);
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
    CHECK_CUDA_ERROR(cudaFree(d_block_flush_to_zero));
    CHECK_CUDA_ERROR(cudaFree(d_block_amax));
    CHECK_CUDA_ERROR(cudaFree(d_block_error));

    CHECK_CUDA_ERROR(cudaFree(d_global_non_zero));
    CHECK_CUDA_ERROR(cudaFree(d_global_flush_to_zero));
    CHECK_CUDA_ERROR(cudaFree(d_global_amax));
    CHECK_CUDA_ERROR(cudaFree(d_global_error));
    ////CHECK_CUDA_ERROR(cudaFree(d_global_error));
}

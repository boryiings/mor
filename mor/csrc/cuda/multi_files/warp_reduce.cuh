#pragma once

#include <cfloat>

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Helper functions for warp-level reduction
__inline__ __device__ float warpReduceMax(float val)
{
    // Perform warp reduction for maximum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceMin(float val) {
    // Perform warp reduction for minimum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    // Perform warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
} 

// Reduction helper function (amax and non-zero counts)
__device__ void blockReduceAmaxNonZero(float local_amax, int local_non_zero, float* s_amax, int* s_non_zero, int flattened_tid) {
    // Step 1: Perform warp-level reduction within each warp
    local_amax = warpReduceMax(local_amax);
    local_non_zero = warpReduceSum(local_non_zero);

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
	local_amax = warpReduceMax(local_amax);
	local_non_zero = warpReduceSum(local_non_zero);
    }

    // Step 4: Write the final value to shared memory, so the final amax/amin are
    // available to all threads.
    if (flattened_tid == 0) {
	s_amax[0] = local_amax;
	s_non_zero[0] = local_non_zero;
    }
    __syncthreads();
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
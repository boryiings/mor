#include "cuda_utils.cuh"

// Function implementation
int get_num_SMs() {
    int device;
    cudaGetDevice(&device);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    return numSMs;
} 
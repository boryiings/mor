


#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include "gpu_computation.cuh"

#include "general_includes.hpp"

#define CHECK_CUDA_ERROR(call)                                   \
    {                                                            \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess)                                  \
        {                                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                             \
        }                                                        \
    }


extern int global_scale_down_factor_e4m3_min;
int Get_Bin_Files(char *dirname, char ***o_Filenames);
void Parse_File(char *filename, fp32 **A, int *M, int *N);

#include <cuda_runtime.h>
#include <stdio.h>


// Forward declaration of the kernel
void Phases1And2(const float* A, fp32 *B, int M, int N, int block_dim_ROWS, int block_dim_COLS);


void PrintGpuCapabilities()
{
    PRINT_GRAY;
    cudaDeviceProp prop;
    int device;

    // Get the current device
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Maximum threads per block
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Maximum dimensions of a block (x, y, z)
    printf("Max block dimensions: x=%d, y=%d, z=%d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    // Maximum dimensions of a grid (x, y, z)
    printf("Max grid dimensions: x=%d, y=%d, z=%d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Shared memory per block
    printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);

    // Total constant memory
    printf("Total constant memory: %ld bytes\n", prop.totalConstMem);

    // Warp size
    printf("Warp size: %d threads\n", prop.warpSize);

    // Max pitch
    printf("Max pitch: %ld bytes\n", prop.memPitch);

    // Max number of registers per block
    printf("Registers per block: %d\n", prop.regsPerBlock);

    // Number of multiprocessors and cores per multiprocessor
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);

    // Maximum memory allocation
    printf("Max memory allocation: %ld bytes\n", prop.totalGlobalMem);

    // Memory clock rate and bus width
    printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);

    // Peak memory bandwidth
    double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("Peak memory bandwidth: %.2f GB/s\n", bandwidth);

    // Check cooperative kernel support
    int cooperativeLaunchSupported;
    cudaDeviceGetAttribute(&cooperativeLaunchSupported, cudaDevAttrCooperativeLaunch, device);
    if (!cooperativeLaunchSupported) {
        printf("Cooperative kernel launch is NOT supported on this device.\n");
        PRINT_RESET;
        return; // Exit as cooperative kernels are not supported
    }
    printf("Cooperative kernel launch is supported on this device.\n");

    // Calculate maximum cooperative grid blocks
#if 0
    int maxBlocksPerSM;
    dim3 blockDim(32, 8); // Example block dimensions
    size_t sharedMemPerBlock = 0; // Example shared memory usage per block

    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        (void*)combinedKernelPhases1And2, // Pass an actual kernel here
        blockDim.x * blockDim.y,
        sharedMemPerBlock
    );

    if (err != cudaSuccess) {
        printf("Error in calculating cooperative grid limits: %s\n", cudaGetErrorString(err));
        PRINT_RESET;
        return;
    }

    int totalMaxCgBlocks = maxBlocksPerSM * prop.multiProcessorCount;
    printf("Max cooperative grid blocks: %d (across all SMs)\n", totalMaxCgBlocks);
#endif

    PRINT_RESET;
}


__global__ void blockMaxKernel(const float* d_matrix, float* d_blockMax, int M, int N) {
    // Compute thread and block indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Global indices
    int row = blockRow * blockDim.y + threadRow;
    int col = blockCol * blockDim.x + threadCol;

    // Shared memory for block reduction
    __shared__ float blockShared[16 * 16]; // Updated for 16x16 threads

    // Linear thread index within the block
    int threadIdxLinear = threadRow * blockDim.x + threadCol;

    // Initialize shared memory
    float val = (row < M && col < N) ? fabsf(d_matrix[row * N + col]) : -INFINITY;
    ///float val = (row < M && col < N) ? (d_matrix[row * N + col]) : -INFINITY;

#if 0
    // Print for debugging
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Thread (Block: [%d, %d], Thread: [%d, %d]), Row: %d, Col: %d, Val: %e\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col, val);
    }
#endif

    blockShared[threadIdxLinear] = val;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride /= 2) {
        if (threadIdxLinear < stride) {
            blockShared[threadIdxLinear] = fmaxf(blockShared[threadIdxLinear], blockShared[threadIdxLinear + stride]);
        }
        __syncthreads();
    }

    // Store block max to global memory
    if (threadIdxLinear == 0) {
        d_blockMax[blockRow * gridDim.x + blockCol] = blockShared[0];
    }
}

__global__ void reduceMaxKernel(const float* d_blockMax, float* d_globalMax, int numBlocks) 
{
    __shared__ float shared[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < numBlocks) ? d_blockMax[idx] : -INFINITY;
    __syncthreads();

    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Write global max
    if (tid == 0) {
        atomicMax((int*)d_globalMax, __float_as_int(shared[0]));
    }
}

fp32 GPU_Compute_Max(const fp32* h_matrix, int M, int N) 
{
    fp32 *d_matrix, *d_blockMax, *d_globalMax;
    fp32 answer = 0;

    // Allocate device memory
    cudaMalloc(&d_matrix, M * N * sizeof(float));
    cudaMalloc(&d_blockMax, ((M + 15) / 16) * ((N + 15) / 16) * sizeof(float));
    cudaMalloc(&d_globalMax, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize global max on device
    float initVal = -INFINITY;
    cudaMemcpy(d_globalMax, &initVal, sizeof(float), cudaMemcpyHostToDevice);


    auto start_gpu = std::chrono::high_resolution_clock::now();
    int max_iterations = 100;

    for(int iter = 0; iter < max_iterations; iter++)
    {
        // Launch block max kernel
        dim3 blockDim(16, 16); // Updated block dimensions
        dim3 gridDim((N + 15) / 16, (M + 15) / 16); // Updated grid dimensions
        blockMaxKernel<<<gridDim, blockDim>>>(d_matrix, d_blockMax, M, N);
        cudaDeviceSynchronize();

        // Launch reduction kernel
        int numBlocks = ((M + 15) / 16) * ((N + 15) / 16);
        reduceMaxKernel<<<(numBlocks + 1023) / 1024, 1024>>>(d_blockMax, d_globalMax, numBlocks);
        cudaDeviceSynchronize();
    }

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> gpu_duration = end_gpu - start_gpu;
    double avg_gpu_time = gpu_duration.count()/max_iterations;

    PRINT_RED;
    printf("------------------------------------------------------------------------------------------------------\n");
    std::cout << ">>>>>>>>>>>>>> Average GPU computation completed! Time (averaged over " << max_iterations << " iterations): "<< avg_gpu_time << " micro-seconds" << std::endl;
    {
        double effective_bandwidth = 0;
        effective_bandwidth = sizeof(float) * (M * N);
        effective_bandwidth /= avg_gpu_time;
        effective_bandwidth *= (1000 * 1000)/1024/1024.0/1024.0;
        printf(">>>>>>>>>>>>>> Effective B/W = %f GB/sec\n", effective_bandwidth);
    }
    printf("------------------------------------------------------------------------------------------------------\n");
    PRINT_RESET;


    // Copy result back to host
    cudaMemcpy(&answer, d_globalMax, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_blockMax);
    cudaFree(d_globalMax);

    return answer;
}


#include "single_kernel_quantization.cu"

void Call_GPU_Benchmark2(char *Data_dirname, int scale_down_factor_e4m3_min, int block_dim_ROWS, int block_dim_COLS)
{
    PrintGpuCapabilities();
    char **Filenames = NULL;
    int number_of_files = 0;

    number_of_files = Get_Bin_Files(Data_dirname, &Filenames);
    printf("number_of_files = %d\n", number_of_files);
    printf("Data_dirname = %s\n", Data_dirname);

    for(int overall_iter = 0; overall_iter < number_of_files; overall_iter++)
    {
        ///////if (overall_iter < 120) continue;
        //if (overall_iter >= 1) continue;
        fp32* A_fp32 = NULL;
        int M = 0, N = 0;

        Parse_File(Filenames[overall_iter], &A_fp32, &M, &N); 


        fprintf(stderr, "overall_iter = %d\n", overall_iter);
        fprintf(stderr, "Working on %s\n", Filenames[overall_iter]);
        fprintf(stderr, "M = %d :: N = %d\n", M, N);
        fprintf(stderr, "=========================\n");

        printf("Working on %s\n", Filenames[overall_iter]);
        printf("M = %d :: N = %d\n", M, N);

        int BLOCK_ROWS = 32, BLOCK_COLS = 32;
        //HETERO *Blocked_A = NULL;
        //int number_of_blocks_A = 0;

        BLOCK_ROWS = 32; BLOCK_COLS = 32;
        BLOCK_ROWS = block_dim_ROWS; BLOCK_COLS = block_dim_COLS;

        if (scale_down_factor_e4m3_min != 1) global_scale_down_factor_e4m3_min = scale_down_factor_e4m3_min;

        if (M < BLOCK_ROWS) BLOCK_ROWS = M;
        if (N < BLOCK_COLS) BLOCK_COLS = N;

#if 0
        float matrix_amax = 0;
        matrix_amax = GPU_Compute_Max(A_fp32, M, N);
        printf("matrix_amax = %e\n", matrix_amax);
#else
        fp32 *B_fp32 = (fp32 *)malloc(M * N * sizeof(fp32));
        Phases1And2(A_fp32, B_fp32, M, N, block_dim_ROWS, block_dim_COLS);
        free(B_fp32);
#endif

        free(A_fp32);
    }

    for(int q = 0; q < number_of_files; q++) free(Filenames[q]);
    free(Filenames);
}

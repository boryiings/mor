

#include <iostream>
#include <chrono>
#include "cpu_computation.hpp"  // CPU function declarations
#include "gpu_computation.cuh"  // GPU function declarations

#include "general_includes.hpp"


void Parse_File(char *filename, fp32 **A, int *M, int *N);
void Call_CPU_Benchmark2(char *Data_dirname, int scale_down_factor_e4m3_min, int block_dim_ROWS, int block_dim_COLS);
void Call_GPU_Benchmark2(char *Data_dirname, int scale_down_factor_e4m3_min, int block_dim_ROWS, int block_dim_COLS);



// Function to compare CPU and GPU results
void CompareOutputs(float* Z_cpu, float* Z_gpu, int number_of_rows, int number_of_cols)
{
    bool match = true;
    int max_number_of_mismatches_to_be_printed = 10;
    int number_of_mismatches_printed = 0;
    int number_of_mismatches_found = 0;

    for (int r = 0; r < number_of_rows; r++) 
    {
        for(int c = 0; c < number_of_cols; c++)
        {
            int indx = GET_INDEX(r, c, number_of_cols);
            ///printf("[%d, %d] --> CPU (%e) Vs GPU (%e)\n", r, c, Z_cpu[indx], Z_gpu[indx]);

            if (fabs(Z_cpu[indx] - Z_gpu[indx]) > 1e-4) 
            {
                match = false;
                number_of_mismatches_found++;
                if (number_of_mismatches_printed < max_number_of_mismatches_to_be_printed)
                {
                    std::cout << "Mismatch at index " << indx << "(" << r << "," << c << ")" << ": Z_cpu = " << Z_cpu[indx] << ", Z_gpu = " << Z_gpu[indx] << std::endl;
                    number_of_mismatches_printed++;
                }
            }
        }
    }

    if (match) std::cout << "CPU and GPU results match!" << std::endl;
    else std::cout << "CPU and GPU results do not match! with " << number_of_mismatches_found << " out of " << number_of_rows * number_of_cols << std::endl;
}


void Free_Memory(void *X)
{
    if (X) free(X);
}


void Initialize(float *X, int xcount)
{
    for(int i = 0; i < xcount; i++) X[i] = (rand() % 11) * 0.1;
}

int main(int argc, char **argv)
{
    srand(95123);

    if (argc != 4)
    {
        printf("Usage: %s <Dir_Name> <DIM_ROWS> <DIM_COLS>\n", argv[0]);
        exit(123);
    }

    int block_dim_ROWS, block_dim_COLS;

    sscanf(argv[2], "%d", &block_dim_ROWS);
    sscanf(argv[3], "%d", &block_dim_COLS);

    fp32 scale_down_factor_e4m3_min = 1;


    Call_GPU_Benchmark2(argv[1], scale_down_factor_e4m3_min, block_dim_ROWS, block_dim_COLS);

    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");

    Call_CPU_Benchmark2(argv[1], scale_down_factor_e4m3_min, block_dim_ROWS, block_dim_COLS);
}


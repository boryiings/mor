
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Include this header for memcpy
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "numeric_types.hpp"

void Compute_Hetero_Representation(fp32* A, int M, int P, int block_rows, int block_cols, HETERO **Matrix_A, int *number_of_blocks_A);

void Parse_File(char *filename, fp32 **A, int *M, int *N)
{
    int vals_read = -1;
    FILE *fp = fopen(filename, "rb");

    vals_read = fread(M, sizeof(int), 1, fp); ASSERT(vals_read == 1);
    vals_read = fread(N, sizeof(int), 1, fp); ASSERT(vals_read == 1);

    *A = (fp32 *)malloc((*M) * (*N) * sizeof(fp32));
    vals_read = fread(*A, sizeof(fp32), (*M) * (*N), fp);
    ASSERT(vals_read == ((*M) * (*N)));

    fclose(fp);
    ///printf("vals_read = %d\n", vals_read);
}

int Get_Bin_Files(char *dirname, char ***o_Filenames)
{
    int number_of_files = 0;
    char **Filenames = NULL;

    char command[1024];

    char rand_filename[512];
    snprintf(rand_filename, 512, "/tmp/file-%d", getpid());

    snprintf(command, 1024, "find %s | grep bin | sort -n > %s", dirname, rand_filename);
    int ret_val = system(command);

    char line[2048];

    FILE *fp = fopen(rand_filename, "r");
    while (!feof(fp))
    {
        line[0] = '\0';
        char *ret_val = fgets(line, 2048, fp);
        if (line[0] == '\0') break;

        line[strlen(line)-1] = '\0';
        number_of_files++;
        Filenames = (char **)realloc(Filenames, number_of_files * sizeof(char **));
        Filenames[number_of_files-1] = (char *)malloc(strlen(line) + 1);
        memcpy(Filenames[number_of_files-1], line, strlen(line) + 1);
    }
    fclose(fp);

    //for(int t = 0; t < number_of_files; t++) printf("[%d] -- %s\n", t, Filenames[t]); exit(123);
    *o_Filenames = Filenames;
    return number_of_files;
}

int global_scale_down_factor_e4m3_min = 1;

void Call_CPU_Benchmark2(char *Data_dirname, int scale_down_factor_e4m3_min, int block_dim_ROWS, int block_dim_COLS)
{
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
        HETERO *Blocked_A = NULL;
        int number_of_blocks_A = 0;

        BLOCK_ROWS = 32; BLOCK_COLS = 32;
        BLOCK_ROWS = block_dim_ROWS; BLOCK_COLS = block_dim_COLS;

        if (scale_down_factor_e4m3_min != 1) global_scale_down_factor_e4m3_min = scale_down_factor_e4m3_min;

        if (M < BLOCK_ROWS) BLOCK_ROWS = M;
        if (N < BLOCK_COLS) BLOCK_COLS = N;

        Compute_Hetero_Representation(A_fp32, M, N, BLOCK_ROWS, BLOCK_COLS, &Blocked_A, &number_of_blocks_A);

        for(int i = 0; i < number_of_blocks_A; i++) Blocked_A[i].Free_Memory();
        Free_Memory(Blocked_A);

        free(A_fp32);
    }

    for(int q = 0; q < number_of_files; q++) free(Filenames[q]);
    free(Filenames);
}





#include <iostream>
#include <cmath>
#include "cpu_computation.hpp"


// Function to perform attention mechanism on CPU
void CPU_Code(float* Q, float* K, float* V, float* Z, int number_of_rows_Q, int number_of_rows_KV, int dim_head) 
{
    std::fill(Z, Z + number_of_rows_Q * dim_head, 0.0f);
    float one_over_sqrt_dim_head = 1.0/sqrt(128);

    for (int q = 0; q < number_of_rows_Q; q++) 
    {
        float e_sum = 0.0f;
        for (int k = 0; k < number_of_rows_KV; k++) 
        {
            float qkT = 0.0f;
            for (int d = 0; d < dim_head; d++) qkT += Q[GET_INDEX(q, d, dim_head)] * K[GET_INDEX(k, d, dim_head)];
            qkT *= one_over_sqrt_dim_head;

            float e_value = expf(qkT);

            for (int d = 0; d < dim_head; d++) Z[GET_INDEX(q, d, dim_head)] += e_value * V[GET_INDEX(k, d, dim_head)];
            e_sum += e_value;
        }

        // Normalize the result for the current Q row
        for (int d = 0; d < dim_head; d++) Z[GET_INDEX(q, d, dim_head)] /= e_sum;
        if (q == 0) printf("e_sum = %e\n", e_sum);
    }
}


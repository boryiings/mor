
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Include this header for memcpy

#include "numeric_types.hpp"

fp32 global_clamp_threshold = 1e-30;
            
void Compute_Min_Max_Etc(fp32 *A, int rows, int cols, fp32 *o_min, fp32 *o_max)
{
    ASSERT(rows > 0); ASSERT(cols > 0);
    fp32 min_here = 0.0, max_here = 0.0;

    int vals_considered_so_far = 0;
    for(int i = 0; i < (rows * cols); i++)
    {
        fp32 abs_value = (A[i] < 0) ? -A[i] : A[i];
        if (max_here < abs_value) max_here = abs_value;
        if (abs_value == 0) continue;

        if (vals_considered_so_far == 0) min_here = abs_value;
        else if (abs_value < min_here) min_here = abs_value;
        vals_considered_so_far++;
    }

    if (max_here == 0) max_here = global_clamp_threshold;

    if (vals_considered_so_far == 0) 
    {
        ASSERT(max_here == 0.0);
        printf("vals_considered_so_far == 0; hence resetting min_here to %e ", global_clamp_threshold);
        max_here = min_here = global_clamp_threshold;
    }

#if 0
    if (min_here < global_clamp_threshold) 
    {
        printf(" { Updating min_here from %e --> %e } ", min_here, global_clamp_threshold);
        min_here = global_clamp_threshold;
    }
#endif

    *o_min = min_here;
    *o_max = max_here;
}

extern int global_scale_down_factor_e4m3_min;

void Convert_FP8_E4M3(fp32 *A, fp8_e4m3 *B, int rows, int cols, fp32 *o_scaling_factor, bool *if_possible, double *local_error, int *local_local_number_of_non_zero_values, int matrix_sf_mantissa)
{
    fp32 max_normal_fp8_e4m3 = 448; //S.1111.110
    fp32 min_normal_fp8_e4m3 = 1/64.0; //S.0001.000

    if (global_scale_down_factor_e4m3_min != 1) 
        min_normal_fp8_e4m3 /= global_scale_down_factor_e4m3_min;

    fp32 golden_ratio = max_normal_fp8_e4m3/min_normal_fp8_e4m3;

    fp32 min_here = 0.0;
    fp32 max_here = 0.0;

    Compute_Min_Max_Etc(A, rows, cols, &min_here, &max_here);

    fp32 ratio_here = max_here/min_here;
    if (ratio_here <= golden_ratio) *if_possible = true; else *if_possible = false;

    fp32 scaling_factor = max_normal_fp8_e4m3/max_here;
    PRINT_RED;
    printf("ratio_here = %.3e (%.3e/%.3e) (GR: %.3e) :: scaling_ftr = %.3e :: clamp_thresh = %.3e ", ratio_here, max_here, min_here, golden_ratio, scaling_factor, global_clamp_threshold);
    PRINT_RESET;

    if (1)
    {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&scaling_factor);
        uint32_t new_sf_bits = (bits & 0xFF800000) | matrix_sf_mantissa;
        fp32 new_scaling_factor = *reinterpret_cast<fp32*>(&new_sf_bits);
        if (new_scaling_factor <= scaling_factor) scaling_factor = new_scaling_factor;
        else                                      scaling_factor = new_scaling_factor/2;

        bits = *reinterpret_cast<uint32_t*>(&scaling_factor);
        uint32_t res_mantissa_bits = bits & 0x007FFFFF;
        printf(":: new_scaling_ftr = %.3e :: scaling_ftr_new = %.3e :: res_mantissa_bits = %d :: ", new_scaling_factor, scaling_factor, res_mantissa_bits);
        ASSERT(res_mantissa_bits == matrix_sf_mantissa);
    }

    if ( 0 && (ratio_here < golden_ratio/2))
    {
        printf("{{ Before sf = %e :: After = ", scaling_factor);
        float log_scaling_factor = log2(scaling_factor);
        log_scaling_factor = floor(log_scaling_factor);
        scaling_factor = pow(2, log_scaling_factor);
        printf("%e }}", scaling_factor);
    }

    fp32 *Tmp_fp32 = (fp32 *)malloc(rows * cols * sizeof(fp32));
    for(int i = 0; i < (rows * cols); i++) Tmp_fp32[i] = A[i] * scaling_factor;
    for(int i = 0; i < (rows * cols); i++) B[i] = fp8_e4m3(Tmp_fp32[i]);
    free(Tmp_fp32);

    {
        fp32 *TT = (fp32 *)malloc(rows * cols * sizeof(fp32));
        for(int i = 0; i < (rows * cols); i++) TT[i] = B[i].toFP32();
        for(int i = 0; i < (rows * cols); i++) TT[i] /= scaling_factor;

        //for(int i = 0; i < (rows * cols); i++) printf("\nZZZ:: A[%d] = %e :: TT[%d] = %e\n", i, A[i], i, TT[i]); 

        int ctr = 0;
        double sum_ratio = 0;
        double delta_sum_sqr = 0.0;
        int number_of_artificial_zeros_created = 0;

        for(int i = 0; i < (rows * cols); i++)
        {
            if (A[i] == 0) { ASSERT(TT[i] == 0); continue;}
            fp32 ratio = (TT[i] - A[i])/A[i];
            if (ratio < 0) ratio = -ratio;
            if (*if_possible)
            {
                if (ratio > .06) 
                    printf("   rattio = %e :: A[i] = %e :: TT[i] = %e   ", ratio, A[i], TT[i]);
            }

            sum_ratio += ratio;
            delta_sum_sqr += (A[i] - TT[i])*(A[i] - TT[i]);
            ctr++;

            if (TT[i] == 0.0) { printf(" {{%e --Q--> %e}} ", A[i], TT[i]); number_of_artificial_zeros_created++; }
        }

        if (ctr == 0) printf("Avg ratio ( fp8_e4m3 ) = %.3e :: %d :: %.3e :: %.3e :: %d\n", 0.0, 0, 0.0, 0.0, number_of_artificial_zeros_created);
        else printf("Avg ratio ( fp8_e4m3 ) = %.3e :: %d :: %.3e :: %.3e :: %d\n", sum_ratio/ctr, ctr, sum_ratio, delta_sum_sqr, number_of_artificial_zeros_created);
        free(TT);

        *local_error = sum_ratio;
        *local_local_number_of_non_zero_values = ctr;
    }

    *o_scaling_factor = scaling_factor;
}

void Convert_FP8_E5M2(fp32 *A, fp8_e5m2 *B, int rows, int cols, fp32 *o_scaling_factor, bool *if_possible,  double *local_error, int *local_local_number_of_non_zero_values, int matrix_sf_mantissa)
{
    fp32 max_normal_fp8_e5m2 = 57344; //S.11110.11
    fp32 min_normal_fp8_e5m2 = 1/16384.0; //S.00001.00
    fp32 golden_ratio = max_normal_fp8_e5m2/min_normal_fp8_e5m2;

    fp32 min_here = 0.0;
    fp32 max_here = 0.0;

    Compute_Min_Max_Etc(A, rows, cols, &min_here, &max_here);

    fp32 ratio_here = max_here/min_here;
    if (ratio_here <= golden_ratio) *if_possible = true; else *if_possible = false;

    fp32 scaling_factor = max_normal_fp8_e5m2/max_here;
    PRINT_RED;
    printf("ratio_here = %.3e (%.3e/%.3e) (GR: %.3e) :: scaling_ftr = %.3e :: clamp_thresh = %.3e ", ratio_here, max_here, min_here, golden_ratio, scaling_factor, global_clamp_threshold);
    PRINT_RESET;

    if (1)
    {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&scaling_factor);
        uint32_t new_sf_bits = (bits & 0xFF800000) | matrix_sf_mantissa;
        fp32 new_scaling_factor = *reinterpret_cast<fp32*>(&new_sf_bits);
        if (new_scaling_factor <= scaling_factor) scaling_factor = new_scaling_factor;
        else                                      scaling_factor = new_scaling_factor/2;


        bits = *reinterpret_cast<uint32_t*>(&scaling_factor);
        uint32_t res_mantissa_bits = bits & 0x007FFFFF;
        printf(":: new_scaling_ftr = %.3e :: scaling_ftr_new = %.3e :: res_mantissa_bits = %d :: ", new_scaling_factor, scaling_factor, res_mantissa_bits);
        ASSERT(res_mantissa_bits == matrix_sf_mantissa);
    }

    fp32 *Tmp_fp32 = (fp32 *)malloc(rows * cols * sizeof(fp32));
    for(int i = 0; i < (rows * cols); i++) Tmp_fp32[i] = A[i] * scaling_factor;
    for(int i = 0; i < (rows * cols); i++) B[i] = fp8_e5m2(Tmp_fp32[i]);
    free(Tmp_fp32);

    {
        fp32 *TT = (fp32 *)malloc(rows * cols * sizeof(fp32));
        for(int i = 0; i < (rows * cols); i++) TT[i] = B[i].toFP32();
        for(int i = 0; i < (rows * cols); i++) TT[i] /= scaling_factor;

        int ctr = 0;
        double sum_ratio = 0;
        double delta_sum_sqr = 0.0;
        int number_of_artificial_zeros_created = 0;

        for(int i = 0; i < (rows * cols); i++)
        {
            if (A[i] == 0) { ASSERT(TT[i] == 0); continue;}
            fp32 ratio = (TT[i] - A[i])/A[i];
            if (ratio < 0) ratio = -ratio;
            sum_ratio += ratio;
            delta_sum_sqr += (A[i] - TT[i])*(A[i] - TT[i]);
            ctr++;
            if (TT[i] == 0.0) { printf(" {{%e --Q--> %e}} ", A[i], TT[i]); number_of_artificial_zeros_created++; }
        }
        
        if (ctr == 0) printf("Avg ratio ( fp8_e5m2 ) = %.3e :: %d :: %.3e :: %.3e :: %d\n", 0.0, 0, 0.0, 0.0, number_of_artificial_zeros_created);
        else printf("Avg ratio ( fp8_e5m2 ) = %.3e :: %d :: %.3e :: %.3e :: %d\n", sum_ratio/ctr, ctr, sum_ratio, delta_sum_sqr, number_of_artificial_zeros_created);
        free(TT);

        *local_error = sum_ratio;
        *local_local_number_of_non_zero_values = ctr;
    }

    *o_scaling_factor = scaling_factor;
}

void Create_Blocks(HETERO *Blocked_A, fp32 *A, int M, int P, int BLOCK_ROWS, int BLOCK_COLS, int total_blocks)
{
    //Step 1: Compute the matrix maximum...
    fp32 matrix_amax = 0;
    int nnz = 0;
    for(int i = 0; i < M; i++) 
    {
        for(int j = 0; j < P; j++) 
        {
            fp32 aval = A[i * P + j];
            if (aval < 0) aval = -aval;
            if (matrix_amax < aval) matrix_amax = aval;
            if (aval != 0) nnz++;
        }
    }

    printf("matrix_amax = %e (Hex: 0x%x)\n", matrix_amax, *reinterpret_cast<uint32_t*>(&matrix_amax));
    printf("nnz = %d\n", nnz);
    //exit(123);

    if (matrix_amax <= 0) ERROR_PRINT();
    uint32_t matrix_sf_mantissa = 0;
    {
        //Let's map it to a pure mantissa. 
        ///matrix_amax = 446;
        fp32 max_normal_fp8_e4m3 = 448; //S.1111.110
        fp32 ratio = max_normal_fp8_e4m3/matrix_amax;
        uint32_t bits = *reinterpret_cast<uint32_t*>(&ratio);
        uint32_t mantissa = bits & 0x007FFFFF;
        printf("Hexadecimal: 0x%x\n", bits);
        printf("matrix_amax = %e :: ratio = %e :: mantissa = %d\n", matrix_amax, ratio, mantissa);
        matrix_sf_mantissa = mantissa;
    }

    double total_error_e4m3 = 0, total_error_e5m2 = 0;
    int number_of_non_zero_values = 0;

    int ctr_A = 0;
    for(int i = 0; i < M; i += BLOCK_ROWS)
    {
        for(int j = 0; j < P; j += BLOCK_COLS)
        {
            Blocked_A[ctr_A].num_rows = BLOCK_ROWS;
            Blocked_A[ctr_A].num_cols = BLOCK_COLS;
            Blocked_A[ctr_A].start_row = i;
            Blocked_A[ctr_A].start_col = j;
            Blocked_A[ctr_A].Values_fp32 = (fp32 *)malloc(BLOCK_ROWS * BLOCK_COLS * sizeof(fp32));

            //We now have the blocks starting address...
            int k = 0;
            for(int i1 = 0; i1 < BLOCK_ROWS; i1++)
            {
                for(int j1 = 0; j1 < BLOCK_COLS; j1++)
                {
                    Blocked_A[ctr_A].Values_fp32[k] = A[(i + i1) * P + (j + j1)];
                    k++;
                }
            }
            ASSERT(k == (BLOCK_ROWS * BLOCK_COLS));

            //Now lets try to create bf16 values...
            Blocked_A[ctr_A].Values_bf16 = (bf16 *)malloc(BLOCK_ROWS * BLOCK_COLS * sizeof(bf16));
            for(int q = 0; q < (BLOCK_ROWS * BLOCK_COLS); q++) Blocked_A[ctr_A].Values_bf16[q] = bf16(Blocked_A[ctr_A].Values_fp32[q]);

            //if ((i == 32) && (j == 32)) printf("FDS");
            
            double local_error_m4m3 = 0, local_error_e5m2 = 0;
            int local_number_of_non_zero_values_e4m3 = 0, local_number_of_non_zero_values_e5m2 = 0;

            //Now lets try to create fp8_e4m3 values...
            printf("[%d %d] fp8_e4m3: ", i, j);
            Blocked_A[ctr_A].Values_fp8_e4m3 = (fp8_e4m3 *)malloc(BLOCK_ROWS * BLOCK_COLS * sizeof(fp8_e4m3));
            Convert_FP8_E4M3(Blocked_A[ctr_A].Values_fp32, Blocked_A[ctr_A].Values_fp8_e4m3, BLOCK_ROWS, BLOCK_COLS, &Blocked_A[ctr_A].scale_fp8_e4m3, &Blocked_A[ctr_A].if_possible_fp8_e4m3, &local_error_m4m3, &local_number_of_non_zero_values_e4m3, matrix_sf_mantissa);
            printf("if_possible = %d :: scale_fp8_e4m3 = %e\n", Blocked_A[ctr_A].if_possible_fp8_e4m3, Blocked_A[ctr_A].scale_fp8_e4m3);
            //if (Blocked_A[ctr_A].if_possible_fp8_e4m3 == false) printf("fdsf");

            //Now lets try to create fp8_e5m2 values...
            printf("[%d %d] fp8_e5m2: ", i, j);
            Blocked_A[ctr_A].Values_fp8_e5m2 = (fp8_e5m2 *)malloc(BLOCK_ROWS * BLOCK_COLS * sizeof(fp8_e5m2));
            Convert_FP8_E5M2(Blocked_A[ctr_A].Values_fp32, Blocked_A[ctr_A].Values_fp8_e5m2, BLOCK_ROWS, BLOCK_COLS, &Blocked_A[ctr_A].scale_fp8_e5m2, &Blocked_A[ctr_A].if_possible_fp8_e5m2, &local_error_e5m2, &local_number_of_non_zero_values_e5m2, matrix_sf_mantissa);
            printf("if_possible = %d :: scale_fp8_e5m2 = %e\n", Blocked_A[ctr_A].if_possible_fp8_e5m2, Blocked_A[ctr_A].scale_fp8_e5m2);

            ASSERT(local_number_of_non_zero_values_e4m3 == local_number_of_non_zero_values_e5m2);
            total_error_e4m3 += local_error_m4m3;
            total_error_e5m2 += local_error_e5m2;
            number_of_non_zero_values += local_number_of_non_zero_values_e4m3;

            ctr_A++;
            printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        }
    }
            
    ASSERT(ctr_A == total_blocks);
    {
        int total_number_of_values = M * P;
        fp32 average_error_e4m3 = 0;
        fp32 average_error_e5m2 = 0;
        if (number_of_non_zero_values) 
        {
            average_error_e4m3 = total_error_e4m3/number_of_non_zero_values;
            average_error_e5m2 = total_error_e5m2/number_of_non_zero_values;
        }
        printf("Overall Stats: Avg error ({E4M3) = %e :: %e :: %d} ::  ({E5M2) = %e :: %e :: %d} :: Total = %d\n", average_error_e4m3, total_error_e4m3, number_of_non_zero_values, average_error_e5m2, total_error_e5m2, number_of_non_zero_values, total_number_of_values);
    }
}


		
void Convert_Tensor_from_fp32_to_bf16(fp32 *X_fp32, bf16 *X_bf16, int num_rows, int num_cols)
{
	for(int q = 0; q < num_rows * num_cols; q++)
	{
		X_bf16[q] = bf16(X_fp32[q]);
	}
}

    
void Compute_Hetero_Representation(fp32* A, int M, int P, int block_rows, int block_cols, HETERO **Blocked_A, int *number_of_blocks_A)
{
    ASSERT((M % block_rows) == 0); ASSERT((P % block_cols) == 0);

    *number_of_blocks_A = (M / block_rows) * (P / block_cols);
    *Blocked_A = (HETERO *)malloc((*number_of_blocks_A) * sizeof(HETERO));

    Create_Blocks(*Blocked_A, A, M, P, block_rows, block_cols, *number_of_blocks_A);
}


#if 0
void Perform_Test(void)
{
	srand(95123);
	int noel = 32 * 32;
	fp32 *A_fp32 = (fp32 *)malloc(noel * sizeof(fp32));

	fp32 fp32_min0 = -1;
	fp32 fp32_max0 = +1;
	fp32 fp32_delta0 = fp32_max0 - fp32_min0;

	fp32 fp32_min1 = -65536;
	fp32 fp32_max1 = -1;
	fp32 fp32_delta1 = fp32_max1 - fp32_min1;

	fp32 fp32_min2 = 1;
	fp32 fp32_max2 = 65536;
	fp32 fp32_delta2 = fp32_max2 - fp32_min2;

	for(int q = 0; q < noel; q++)
	{
		fp32 ratio = rand()/(float)(RAND_MAX);
		fp32 delta = fp32_delta0;
		fp32 min = fp32_min0;
		fp32 max = fp32_max0;

		fp32 value = min + ratio * delta;
		A_fp32[q] = value;
	}

	bool set = false;
	fp32 abs_min_fp32, abs_max_fp32;
	for(int q = 0; q < noel; q++)
	{
		if (A_fp32[q] == 0) continue;
		fp32 abs_value = (A_fp32[q] < 0) ? -A_fp32[q]: A_fp32[q];
		if (set == false) { abs_min_fp32 = abs_max_fp32 = abs_value; set = true; }
		else
		{
			if (abs_min_fp32 > abs_value) abs_min_fp32 = abs_value;
			else if (abs_max_fp32 < abs_value) abs_max_fp32 = abs_value;
		}
		ASSERT(set);
	}

	fp32 ratio_max_min = abs_max_fp32/abs_min_fp32;

	int delta_exponent_ratio = 0;
	{
		fp32 f = ratio_max_min;
		uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
		int32_t exponent = ((bits >> 23) & 0xFF);
		delta_exponent_ratio = exponent - 127;
        delta_exponent_ratio++;
	}

	printf("abs_max_fp32 = %f :: abs_min_fp32 = %f :: ratio_max_min = %f :: delta_exponent_ratio = %d\n", abs_max_fp32, abs_min_fp32, ratio_max_min, delta_exponent_ratio);

	fp32 delta_exponent_fp8_e4m3 = 14; //7 - (-7)
	fp32 delta_exponent_fp8_e5m2 = 30; // 15 - (-15)
                                       //

    if (1 & (delta_exponent_ratio <= delta_exponent_fp8_e4m3))
    {
        printf("We can represent data using FP8_E4M3\n");
        for(int q = 0; q < noel; q++)
        {
            fp32 original_val = A_fp32[q];
            fp32 ratio = original_val/abs_max_fp32;
            fp8_e4m3 ratio_fm8_e4m3 = fp8_e4m3(ratio);
            fp32 ratio_reconstructed_fp32 = ratio_fm8_e4m3.toFP32();
            fp32 reconstructed_val = ratio_reconstructed_fp32 * abs_max_fp32;

            uint32_t fp32_bits_0, fp32_bits_1, fp32_bits_2, fp32_bits_3;
		    memcpy(&fp32_bits_0, &(original_val), sizeof(fp32));
		    memcpy(&fp32_bits_1, &(reconstructed_val), sizeof(fp32));
		    memcpy(&fp32_bits_2, &ratio, sizeof(fp32));
		    memcpy(&fp32_bits_3, &ratio_reconstructed_fp32, sizeof(fp32));

            printf("original_val = %f (0x%X) :: reconstructed_val = %f (0x%X) :: ratio = %f (0x%X) :: ratio_reconstructed_fp32 = %f (0x%X)\n", 
                    original_val, fp32_bits_0, reconstructed_val, fp32_bits_1, 
                    ratio, fp32_bits_2, ratio_reconstructed_fp32, fp32_bits_3);
        }
    }
    else if (delta_exponent_ratio <= delta_exponent_fp8_e5m2)
    {
        printf("We can represent data using FP8_E5M2\n");
        for(int q = 0; q < noel; q++)
        {
            fp32 original_val = A_fp32[q];
            fp32 ratio = original_val/abs_max_fp32;
            fp8_e5m2 ratio_fm8_e5m2 = fp8_e5m2(ratio);
            fp32 ratio_reconstructed_fp32 = ratio_fm8_e5m2.toFP32();
            fp32 reconstructed_val = ratio_reconstructed_fp32 * abs_max_fp32;

            uint32_t fp32_bits_0, fp32_bits_1, fp32_bits_2, fp32_bits_3;
		    memcpy(&fp32_bits_0, &(original_val), sizeof(fp32));
		    memcpy(&fp32_bits_1, &(reconstructed_val), sizeof(fp32));
		    memcpy(&fp32_bits_2, &ratio, sizeof(fp32));
		    memcpy(&fp32_bits_3, &ratio_reconstructed_fp32, sizeof(fp32));

            printf("original_val = %f (0x%X) :: reconstructed_val = %f (0x%X) :: ratio = %f (0x%X) :: ratio_reconstructed_fp32 = %f (0x%X)\n", 
                    original_val, fp32_bits_0, reconstructed_val, fp32_bits_1, 
                    ratio, fp32_bits_2, ratio_reconstructed_fp32, fp32_bits_3);

        }
    }

	for(int q = 0; q < 10; q++)
	{
		uint32_t fp32_bits_1;
		memcpy(&fp32_bits_1, A_fp32 + q, sizeof(fp32));

		printf("A_fp32[%d] = %f (0x%X)\n", q, A_fp32[q], fp32_bits_1);
	}
	exit(123);
}
#endif


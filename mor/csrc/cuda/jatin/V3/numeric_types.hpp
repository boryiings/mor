#ifndef _NUMERIC_TYPES_HPP_
#define _NUMERIC_TYPES_HPP_

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h> // Include this header for memcpy
		    
#define PRINT_BLACK         printf("\033[22;30m");
#define PRINT_RED           printf("\033[22;31m");
#define PRINT_GREEN         printf("\033[22;32m");
#define PRINT_BROWN         printf("\033[22;33m");
#define PRINT_BLUE          printf("\033[22;34m");
#define PRINT_MAGENTA       printf("\033[22;35m");
#define PRINT_CYAN          printf("\033[22;36m");
#define PRINT_GRAY          printf("\033[22;37m");
#define PRINT_LIGHT_RED     printf("\033[01;31m");
#define PRINT_LIGHT_GREEN   printf("\033[01;32m");
#define PRINT_RESET         printf("\033[0m");
#define PRINT_BOLD          printf("\033[1m");
#define PRINT_LIGHT         printf("\033[2m");
#define PRINT_ITALICS       printf("\033[3m");
#define PRINT_UNDERLINE     printf("\033[4m");
#define PRINT_SYMBOL        {unsigned char X[4]; X[0] = 226; X[1] = 155; X[2] = 179; X[3] = '\0'; printf("%s", X);}
#define PRINT_SYMBOL2       {unsigned char X[4]; X[0] = 226; X[1] = 155; X[2] = 148; X[3] = '\0'; printf("%s", X);}
#define PRINT_SYMBOL3       {unsigned char X[4]; X[0] = 226; X[1] = 155; X[2] = 142; X[3] = '\0'; printf("%s", X);}
#define PRINT_SYMBOL4       {unsigned char X[4]; X[0] = 226; X[1] = 155; X[2] = 145; X[3] = '\0'; printf("%s", X);}
#define PRINT_SYMBOL5       {unsigned char X[4]; X[0] = 226; X[1] = 155; X[2] = 169; X[3] = '\0'; printf("%s", X);}

#define ERROR_PRINT() { printf("Error in (%s) on line (%d)\n", __FILE__, __LINE__); exit(123);}
#define ASSERT(x) { if (!(x)) ERROR_PRINT(); }

#define NV_ABS(x)           (((x) < 0) ? (-(x)) : (x))

#define fp32 float

void Allocate_Memory(void **A, int number_of_rows, int number_of_cols, int sizeof_element);
void Free_Memory(void *X);

// BF16 format
struct bf16
{
    uint16_t value;

    bf16(fp32 f) //S1 E8 M7
    {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
        uint32_t sign = (bits >> 31) & 0x1;
        int32_t exponent = ((bits >> 23) & 0xFF); // E8
        uint32_t mantissa = (bits >> 16) & 0x7F;
        uint32_t extra_mantissa = (bits >> 15) & 0x1;

        ASSERT(((extra_mantissa == 0) || (extra_mantissa == 1)));
        if (extra_mantissa == 1)
        {
            mantissa++;
            if (mantissa == 128) 
            {
                exponent++;
                mantissa = 0;
            }
        }

        if (exponent == 256) ERROR_PRINT();

	    value = (sign << 15) | (exponent << 7) | mantissa;
    }

    fp32 toFP32() 
    {
        uint32_t sign = (value >> 15) & 0x1;
        int32_t exponent = ((value >> 7) & 0xFF);
        uint32_t mantissa = value & 0x7F;

        uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 16);
        return *reinterpret_cast<float*>(&bits);
    }

};

// FP8 E4M3 format
struct fp8_e4m3 
{
    uint8_t value;

    fp8_e4m3(fp32 f) 
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
            //PRINT_LIGHT_RED;
            //printf(" {{ (fp8_e4m3) nett_exp = %d :: mantis = %d :: f2 = %.3e }} ", exponent, mantissa, f2);
            //PRINT_RESET;
        }
        else
        {
            uint32_t extra_mantissa = (bits >> 19) & 0x1;
            ASSERT(((extra_mantissa == 0) || (extra_mantissa == 1)));
            if (extra_mantissa == 1)
            {
                mantissa++;
                if (mantissa == 8) 
                {
                    exponent++;
                    mantissa = 0;
                }
            }

            if (exponent == 16) ERROR_PRINT();

            if (exponent <= 0)  // Subnormal numbers
            {
                exponent = 0;
                mantissa = (bits >> 23) & 0x7;
            } 
            else if (exponent > 15) 
            { // Infinity or NaN
                exponent = 15;
                mantissa = 7;
                printf("Found a NaN NaN NaN Nan\n");
                ERROR_PRINT();
            }
            else if (exponent == 15)
            {
                if (mantissa == 7)
                {
                    printf("Found a NaN NaN NaN Nan\n");
                    ERROR_PRINT();
                }
            }
        }

        value = (sign << 7) | (exponent << 3) | mantissa;
    }

    fp32 toFP32() 
    {
        uint32_t sign = (value >> 7) & 0x1;
        int32_t exponent = ((value >> 3) & 0xF) - 7 + 127;
        uint32_t mantissa = value & 0x7;

        int32_t just_exponent = ((value >> 3) & 0xF);
        if (just_exponent == 0) //sub_normal case...
        {
            fp32 ret_val = 1.0;
            if (sign == 1) ret_val = -1.0;
            ret_val *= (1/8.0 * 1/64.0); //1/2^6
            ret_val *= mantissa;

            //PRINT_CYAN;
            //printf(" {{ (fp8_e4m3) [%d] --> just_exponent = %d :: sign = %d :: exponent = %d :: mantissa = %d :: ret_val = %e }} ", value, just_exponent, sign, exponent, mantissa, ret_val);
            //PRINT_RESET;

            return ret_val;
        }

        uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 20);
        return *reinterpret_cast<float*>(&bits);
    }
};

// FP8 E5M2 format
struct fp8_e5m2 {
    uint8_t value;

    fp8_e5m2(float f) 
    {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
        uint32_t sign = (bits >> 31) & 0x1;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15; // E5 exponent bias is 15
        uint32_t mantissa = (bits >> 21) & 0x3;

        if (exponent <= 0)
        {
            //Dealing with sub_normal...
            fp32 f2 = (f < 0) ? -f : f;
            exponent = 0;

            if (f2 < (1/8.0 * 1/16384.0)) mantissa = 0;
            else if (f2 < (3/8.0 * 1/16384.0)) mantissa = 1;
            else if (f2 < (5/8.0 * 1/16384.0)) mantissa = 2;
            else if (f2 < (7/8.0 * 1/16384.0)) mantissa = 3;
            else
            {
                exponent = 1;
                mantissa = 0;
            }
            //PRINT_LIGHT_RED;
            //printf("{{ (fp8_e5m2) nett_exp = %d :: mantis = %d :: f2 = %.3e }}", exponent, mantissa, f2);
            //PRINT_RESET;
        }
        else
        {
            uint32_t extra_mantissa = (bits >> 20) & 0x1;
 
            ASSERT(((extra_mantissa == 0) || (extra_mantissa == 1)));
            if (extra_mantissa == 1)
            {
                mantissa++;
                if (mantissa == 4) 
                {
                    exponent++;
                    mantissa = 0;
                }
            }

            if (exponent == 32) ERROR_PRINT();

            if (exponent <= 0)  // Subnormal numbers
            {
                exponent = 0;
                mantissa = (bits >> 23) & 0x3;
            } 
            else if (exponent >= 31) 
            { // Infinity or NaN
                exponent = 31;
                mantissa = 0;
            }
        }
        
        value = (sign << 7) | (exponent << 2) | mantissa;
    }

    float toFP32() 
    {
        uint32_t sign = (value >> 7) & 0x1;
        int32_t exponent = ((value >> 2) & 0x1F) - 15 + 127;
        uint32_t mantissa = value & 0x3;

        int32_t just_exponent = ((value >> 2) & 0x1F);
        if (just_exponent == 0) //sub_normal case...
        {
            fp32 ret_val = 1.0;
            if (sign == 1) ret_val = -1.0;
            ret_val *= (1/4.0 * 1/16384.0); //1/2^14
            ret_val *= mantissa;

            //PRINT_CYAN;
            //printf(" {{ (fp8_e5m2) [%d] --> just_exponent = %d :: sign = %d :: exponent = %d :: mantissa = %d :: ret_val = %e }} ", value, just_exponent, sign, exponent, mantissa, ret_val);
            //PRINT_RESET;

            return ret_val;
        }

        uint32_t bits = (sign << 31) | (exponent << 23) | (mantissa << 21);
        return *reinterpret_cast<float*>(&bits);
    }
};

struct HETERO
{
    int num_rows;
    int num_cols;
    int start_row = 0;
    int start_col = 0;

    fp32 *Values_fp32 = NULL;
    bf16 *Values_bf16 = NULL;
    fp8_e4m3 *Values_fp8_e4m3 = NULL;
    fp8_e5m2 *Values_fp8_e5m2 = NULL;

    fp32 scale_fp8_e4m3 = 1.0;
    fp32 scale_fp8_e5m2 = 1.0;

    bool if_possible_fp8_e4m3 = false;
    bool if_possible_fp8_e5m2 = false;

    void Free_Memory(void) 
    {
        if (Values_fp32) free(Values_fp32);
        if (Values_bf16) free(Values_bf16);
        if (Values_fp8_e4m3) free(Values_fp8_e4m3);
        if (Values_fp8_e5m2) free(Values_fp8_e5m2);
    }
};


//bf16 float_to_bf16(fp32 f);
//float bf16_to_float(bf16 bf);

void Convert_Tensor_from_fp32_to_bf16(fp32 *A_fp32, bf16 *A_bf16, int num_rows, int num_cols);
#endif

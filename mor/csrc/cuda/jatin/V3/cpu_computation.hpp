

#ifndef CPU_COMPUTATION_HPP
#define CPU_COMPUTATION_HPP

#define GET_INDEX(i, j, number_of_cols) ((i) * number_of_cols + (j))

// CPU function declaration
void CPU_Code(float* Q, float* K, float* V, float* Z, int number_of_rows_Q, int number_of_rows_KV, int dim_head);




#endif


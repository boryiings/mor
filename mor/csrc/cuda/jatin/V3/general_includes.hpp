
#ifndef _GENERAL_INCLUDES_HPP_
#define _GENERAL_INCLUDES_HPP_

#define ERROR_PRINT() { printf("Error on line (%d) in file (%s)\n", __LINE__, __FILE__); exit(123); }
#define ASSERT(x) if (!(x)) ERROR_PRINT(); 

#define GET_INDEX(i, j, number_of_cols) ((i) * number_of_cols + (j))

#define fp32 float

#if 1

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
#define PRINT_RESET printf("\033[0m");

#else

#define PRINT_BLACK
#define PRINT_RED
#define PRINT_GREEN
#define PRINT_BROWN
#define PRINT_BLUE
#define PRINT_MAGENTA
#define PRINT_CYAN
#define PRINT_GRAY
#define PRINT_LIGHT_RED
#define PRINT_LIGHT_GREEN
#define PRINT_RESET


#endif




#endif

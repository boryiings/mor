

#!/bin/bash

# Script to compile the project with optimization flags

# Print message indicating the start of the compilation process
echo "Starting optimized compilation of the Quantization Computation project..."

# Compile CPU computation file with optimization
echo "Compiling CPU computation file (cpu_computation.cpp and numeric_types.cpp) with optimizations..."
g++ -O3 -march=native -c cpu_computation.cpp -o cpu_computation.o
g++ -O3 -march=native -c numeric_types.cpp -o numeric_types.o
if [ $? -ne 0 ]; then
    echo "Error: Compilation of cpu_computation.cpp failed!"
    exit 1
fi
echo "Successfully compiled cpu_computation.cpp to cpu_computation.o with optimizations."

# Compile GPU computation file with PTX output
echo "Compiling GPU computation file (gpu_computation.cu) to generate PTX code..."
nvcc -O3 -arch=sm_80 --use_fast_math -ptx gpu_computation.cu -o gpu_computation.ptx
if [ $? -ne 0 ]; then
    echo "Error: Compilation of gpu_computation.cu to PTX failed!"
    exit 1
fi
echo "Successfully generated PTX code in 'gpu_computation.ptx'."

# Compile GPU computation file with SASS output
echo "Compiling GPU computation file (gpu_computation.cu) to generate SASS code..."
nvcc -O3 -arch=sm_80 --use_fast_math -c gpu_computation.cu -o gpu_computation.o --ptxas-options=-v
if [ $? -ne 0 ]; then
    echo "Error: Compilation of gpu_computation.cu to SASS failed!"
    exit 1
fi
echo "Successfully compiled gpu_computation.cu to SASS."

# Compile and link everything together with optimization
echo "Compiling and linking main.cu with the object files with optimizations..."
nvcc -O3 -arch=sm_80 --use_fast_math main.cu gpu_computation.o cpu_computation.o numeric_types.o -o quantization

if [ $? -ne 0 ]; then
    echo "Error: Linking and compilation of main.cu failed!"
    exit 1
fi
echo "Successfully linked and compiled everything into 'quantization' executable with optimizations."

# Final message indicating successful compilation
echo "Optimized Quantization Computation project is ready to be executed as './quantization'."


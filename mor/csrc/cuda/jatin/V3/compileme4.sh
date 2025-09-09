#!/bin/bash

# Script to compile the project with debug flags for PTX and SASS analysis

# Print message indicating the start of the compilation process
echo "Starting debug compilation of the Quantization Computation project..."

# Compile CPU computation file with debugging
echo "Compiling CPU computation file (cpu_computation.cpp and numeric_types.cpp) with debug symbols..."
g++ -O0 -g -march=native -c cpu_computation.cpp -o cpu_computation.o
g++ -O0 -g -march=native -c numeric_types.cpp -o numeric_types.o
if [ $? -ne 0 ]; then
    echo "Error: Compilation of cpu_computation.cpp failed!"
    exit 1
fi
echo "Successfully compiled cpu_computation.cpp to cpu_computation.o with debug symbols."

# Compile GPU computation file with PTX and SASS output
echo "Compiling GPU computation file (gpu_computation.cu) with debug symbols and PTX generation..."
nvcc -O0 -g -lineinfo -arch=sm_80 --use_fast_math -ptx gpu_computation.cu -o gpu_computation.ptx
if [ $? -ne 0 ]; then
    echo "Error: Compilation of gpu_computation.cu to PTX failed!"
    exit 1
fi
echo "Successfully generated PTX code in 'gpu_computation.ptx'."

# Generate SASS code with verbose information
echo "Compiling GPU computation file (gpu_computation.cu) with SASS output..."
nvcc -O0 -g -lineinfo -arch=sm_80 --use_fast_math -c gpu_computation.cu -o gpu_computation.o --ptxas-options=-v
if [ $? -ne 0 ]; then
    echo "Error: Compilation of gpu_computation.cu to SASS failed!"
    exit 1
fi
echo "Successfully compiled gpu_computation.cu to SASS with verbose output."

# Compile and link everything together with debugging
echo "Compiling and linking main.cu with the object files with debug symbols..."
nvcc -O0 -g -lineinfo -arch=sm_80 --use_fast_math main.cu gpu_computation.o cpu_computation.o numeric_types.o -o quantization_debug

if [ $? -ne 0 ]; then
    echo "Error: Linking and compilation of main.cu failed!"
    exit 1
fi
echo "Successfully linked and compiled everything into 'quantization_debug' executable with debug symbols."

# Final message indicating successful compilation
echo "Debug-enabled Quantization Computation project is ready to be executed as './quantization_debug'."

# Optional commands to view PTX or SASS code
echo "Use the following commands to analyze the output:"
echo "  View PTX: more gpu_computation.ptx"
echo "  Analyze SASS: nvdisasm gpu_computation.o"


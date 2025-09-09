

#!/bin/bash

# Script to compile the project without fast math and with optimization flags

# Print message indicating the start of the compilation process
echo "Starting compilation of the Quantization Computation project without fast math..."

# Compile CPU computation file
echo "Compiling CPU computation file (cpu_computation.cpp and numeric_types.cpp)..."
g++ -O3 -march=native -c cpu_computation.cpp -o cpu_computation.o
g++ -O3 -march=native -c numeric_types.cpp -o numeric_types.o
if [ $? -ne 0 ]; then
    echo "Error: Compilation of cpu_computation.cpp failed!"
    exit 1
fi
echo "Successfully compiled cpu_computation.cpp to cpu_computation.o."

# Compile GPU computation file
echo "Compiling GPU computation file (gpu_computation.cu) without fast math..."
nvcc -O3 -arch=sm_80 -c gpu_computation.cu -o gpu_computation.o
##nvcc -O3 -arch=sm_86 -Xptxas -dlcm=ca -maxrregcount=32 -c gpu_computation.cu -o gpu_computation.o
if [ $? -ne 0 ]; then
    echo "Error: Compilation of gpu_computation.cu failed!"
    exit 1
fi
echo "Successfully compiled gpu_computation.cu to gpu_computation.o."

# Compile and link everything together
echo "Compiling and linking main.cu with the object files..."
nvcc -O3 -arch=sm_80 main.cu gpu_computation.o cpu_computation.o numeric_types.o -o quantization

if [ $? -ne 0 ]; then
    echo "Error: Linking and compilation of main.cu failed!"
    exit 1
fi
echo "Successfully linked and compiled everything into 'quantization' executable."

# Final message indicating successful compilation
echo "Quantization Computation project is ready to be executed as './quantization'."


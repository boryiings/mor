

g++ -g -O0 -c cpu_computation.cpp -o cpu_computation.o
g++ -g -O0 -c numeric_types.cpp -o numeric_types.o
nvcc -g -G -arch=sm_80 -lineinfo -c gpu_computation.cu -o gpu_computation.o
nvcc -g -G -arch=sm_80 -lineinfo main.cu gpu_computation.o cpu_computation.o numeric_types.o -o quantization

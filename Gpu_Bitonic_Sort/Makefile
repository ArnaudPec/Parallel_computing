max:
	nvcc -c milli.c find_max.cu
	nvcc -o max milli.o find_max.o

bitonic : 
	nvcc -c milli.c bitonic_kernel.cu bitonic_main.cpp
	nvcc -o bitonic bitonic_kernel.o bitonic_main.o

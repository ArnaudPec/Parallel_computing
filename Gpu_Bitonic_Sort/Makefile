max:
	nvcc -c milli.c find_max.cu
	nvcc -o max milli.o find_max.o

bitonic : 
	 /usr/local/cuda/bin/nvcc -c milli.c bitonic_kernel.cu bitonic_main.cpp
	 /usr/local/cuda/bin/nvcc -o bitonic bitonic_kernel.o bitonic_main.o milli.o
clean :
	rm -f *o bitonic 

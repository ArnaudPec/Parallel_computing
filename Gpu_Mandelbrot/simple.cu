// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int N = 16; 
const int blocksize = 16; 

__global__ void parallel_sqrt(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);
	
	for (int i = 0; i < N; i++)
        c[i]=16;

	cudaMalloc( (void**)&cd, size );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

    cudaMemcpy(cd,c,size,cudaMemcpyHostToDevice);

	simple<<<dimGrid, dimBlock>>>(cd);

    cudaThreadSynchronize();

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	
	printf("\ndone\n");

	cudaFree( cd );
	delete[] c;

    return EXIT_SUCCESS;
}

/*
1. 16 cores on 1 SM.

2. Yes. However, before Cuda, in early GPU computing days it could
be difficult to know if floating point are our particular GPU 
( different architectures), if they even handle floating point
operations.

   */


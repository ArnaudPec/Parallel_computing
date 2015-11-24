#include <stdio.h>

__global__ void addMatrix(float *a, float *b, float *c){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = idx+idy;
    c[offset]=a[offset]+b[offset];
}

int main(){
	
	const int N = 16;

	float a[N*N];
	float b[N*N];
	float c[N*N];
    int size = N*N*sizeof(float);
    float *cd;
    float *ad;
    float *bd;
    float theTime;

    cudaEvent_t myEvent, laterEvent;
    cudaEventCreate(&myEvent);
    cudaEventCreate(&laterEvent);
    cudaEventRecord(myEvent, 0);

    cudaMalloc((void **)&cd, size); 
    cudaMalloc((void **)&ad, size); 
    cudaMalloc((void **)&bd, size); 

    cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(cd,c,size,cudaMemcpyHostToDevice);

	dim3 dimBlock( N, N );
	dim3 dimGrid( 1, 1 );

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

	addMatrix <<< dimGrid, dimBlock >>>(ad,bd,cd);
    cudaThreadSynchronize();

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

    cudaEventRecord(laterEvent, 0);
    cudaEventSynchronize(laterEvent); 
    cudaEventElapsedTime(&theTime, myEvent, laterEvent);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}

	printf("%3.1f ms\n", theTime);

    cudaEventDestroy(&laterEvent);
    cudaEventDestroy(&myEvent);

    cudaFree(cd);
    cudaFree(ad);
    cudaFree(bd);

    return 0;
}

/*
 @see addMatrix

 2. It compiles, but it doesn't run because of hw limitations(nowadays std is
 1024 threads per block)
   */

#include <stdio.h>
#define N 22*40

__global__ void addMatrix(float *a, float *b, float *c){

    long idx = blockIdx.x*blockDim.x+threadIdx.x;
    long idy = blockIdx.y*blockDim.y+threadIdx.y;
    //long offset = idx+idy * N;
    long offset = idy+idx * N;
    c[offset]=a[offset]+b[offset];
}

int main(){
	

	float a[N*N];
	float b[N*N];
	float c[N*N];
    long size = N*N*sizeof(float);
    float *cd;
    float *ad;
    float *bd;
    float theTime;

    cudaEvent_t myEvent, laterEvent;
    cudaEventCreate(&myEvent);
    cudaEventCreate(&laterEvent);

    cudaMalloc((void **)&cd, size); 
    cudaMalloc((void **)&ad, size); 
    cudaMalloc((void **)&bd, size); 

	for (long i = 0; i < N; i++)
		for (long j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
    cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(cd,c,size,cudaMemcpyHostToDevice);

	dim3 dimBlock( 22, 22);
	dim3 dimGrid( N/22, N/22 );
/*
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", a[i+j*N]);
		}
		printf("\n");
	}

    printf("\n");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", b[i+j*N]);
		}
		printf("\n");
	}

    printf("\ngpu calc\n");
*/
    cudaEventRecord(myEvent, 0);
	addMatrix <<< dimGrid, dimBlock >>>(ad,bd,cd);
    cudaThreadSynchronize();

    cudaEventRecord(laterEvent, 0);
    
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

    cudaEventSynchronize(laterEvent); 
    cudaEventElapsedTime(&theTime, myEvent, laterEvent);

	for (long i = 0; i < N; i++)
	{
		for (long j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}

	printf("%f ms\n", theTime);

    cudaEventDestroy(laterEvent);
    cudaEventDestroy(myEvent);

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

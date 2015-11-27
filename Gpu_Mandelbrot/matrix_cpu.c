// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"

void add_matrix(float *a, float *b, float *c, int N)
{
	int index, i,j;
	
		for (j = 0; j < N; j++)
	for (i = 0; i < N; i++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int N = 880;

	float a[N*N];
	float b[N*N];
	float c[N*N];
    int start, stop, i,j;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

   start = GetMilliseconds();
	add_matrix(a, b, c, N);
  stop = GetMilliseconds();
/*	
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
*/
	printf("%i ms\n", (stop-start));

}

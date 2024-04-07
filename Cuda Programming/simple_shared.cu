#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define SHMEM_SIZE (2)

__global__ void matrixMul()
{
    __shared__ int A[SHMEM_SIZE];
    A[0] = blockIdx.x + blockIdx.y;
    printf("X=%d,Y=%d Thread ID=%d Info=%d\n", blockIdx.x, blockDim.y, threadIdx.x, A[0]);
}

int main()
{

    dim3 THREADS(2, 2);
    matrixMul<<<THREADS, 2>>>();
}
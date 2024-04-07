#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define SHMEM_SIZE (16 * 16)

void init_matrix(int *m, int N)
{
    for (int i = 0; i < (N * N); i++)
    {
        m[i] = rand() % 10 + 1;
        ;
    }
}

void display_matrix(const int *m, int N)
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %5d", m[(i * N) + j]);
        }

        printf(" \n");
    }
}

void display_transpose(const int *m, int N)
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %5d", m[(j * N) + i]);
        }

        printf(" \n");
    }
}

void verify_result(int *a, int *b, int *c, int N)
{
    int tmp = 0;
    // For every row...
    for (int i = 0; i < N; i++)
    {
        // For every column...
        for (int j = 0; j < N; j++)
        {
            // For every element in the row-column pair
            tmp = 0;
            for (int k = 0; k < N; k++)
            {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
    }
}

__global__ void matrixMul(int *a, int *b, int *c, int N)
{
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    int tmp = 0;
    for (int i = 0; i < ((N + dim - 1) / dim); i++)
    {
        A[ty * dim + tx] = a[(row * N) + (i * dim) + tx];
        B[ty * dim + tx] = b[(i * dim * N) + (ty * N) + col];

        __syncthreads();

        for (int j = 0; j < dim; j++)
        {
            tmp += A[ty * dim + j] * B[j * dim + tx];
        }
        __syncthreads();
    }

    c[row * N + col] = tmp;
}

int main()
{
    int N = 4;

    size_t bytes = N * N * sizeof(int);

    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    init_matrix(a, N);
    init_matrix(b, N);

    int threads = 2;
    int blocks = (N + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    verify_result(a, b, c, N);

    printf("MATRIX A: \n");
    display_matrix(a, N);
    printf(" \n");
    printf("MATRIX B: \n");
    display_matrix(b, N);
    printf("Transpose B: \n");
    display_transpose(b, N);

    printf(" \n");
    printf("MATRIX C: \n");
    display_matrix(c, N);
    return 0;
    // display_matrix(a, N);
}
#include <stdio.h>

// Define matrix dimensions
#define N 3

// Kernel function for matrix addition
__global__ void matrixAdd(float *a, float *b, float *c)
{
    printf("Block x=%d y=%d \n", blockIdx.x, blockIdx.y);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * N + col;

    if (row < N && col < N)
    {
        c[row * N + col] = a[row * N + col] + b[row * N + col];
    }
}

void display_matrix(float MATRIX[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%05.2f ", MATRIX[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    float a[N][N], b[N][N], c[N][N];
    float *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(float);

    // Initialize matrices a and b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = (i + 1) * (j + 1);
            b[i][j] = (i * N) + j;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(2, 2);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    printf("Grid Dim: x=%d,y=%d,y=%d \n", gridDim.x, gridDim.y, gridDim.z);
    // Launch kernel
    matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix A:\n");
    display_matrix(a);

    printf("Matrix B:\n");
    display_matrix(b);

    printf("Matrix C:\n");
    display_matrix(c);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

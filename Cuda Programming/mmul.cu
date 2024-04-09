// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size
const int N = 4;
const int SHMEM_SIZE = 4;

__global__ void matrixMul(const int *a, const int *b, int *c)
{
  int block_idx_x = blockIdx.x;
  int block_idx_y = blockIdx.y;

  int thread_idx_x = threadIdx.x;
  int thread_idx_y = threadIdx.y;

  int row = block_idx_y * blockDim.y + thread_idx_y;
  int col = block_idx_x * blockDim.x + thread_idx_x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x)
  {
    int s_row_index = thread_idx_y * blockDim.x;
    int s_col_index = thread_idx_x;

    int a_row_index = row * N;
    int a_col_index = i + thread_idx_x;

    int b_row_index = i * N + thread_idx_y * N;
    int b_col_index = col;

    s_a[s_row_index + s_col_index] = a[a_row_index + a_col_index];
    s_b[s_row_index + s_col_index] = b[b_row_index + b_col_index];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++)
    {
      int sa_col_idx = j;
      int sb_row_idx = j * blockDim.x;
      tmp += s_a[s_row_index + sa_col_idx] * s_b[sb_row_idx + s_col_index];
    }

    __syncthreads();
  }

  c[row * N + col] = tmp;
}

void display_matrix(const int *matrix)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf(" %3d", matrix[(i * N) + j]);
    }
    printf(" \n");
  }
}

void display_transposed(const int *matrix)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf(" %3d", matrix[(j * N) + i]);
    }
    printf(" \n");
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c)
{
  // For every row...
  for (int i = 0; i < N; i++)
  {
    // For every column...
    for (int j = 0; j < N; j++)
    {
      // For every element in the row-column pair
      int tmp = 0;
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

int main()
{
  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []()
           { return rand() % 10; });
  generate(h_b.begin(), h_b.end(), []()
           { return rand() % 10; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 2;

  int BLOCKS = 2;
  // Blocks per grid dimension (assumes THREADS divides N evenly)
  // int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result
  // verify_result(h_a, h_b, h_c);

  printf(" \n Matrix A \n");
  display_matrix(h_a.data());
  printf(" \n");
  printf("Matrix B \n");
  display_matrix(h_b.data());
  printf(" \n");
  printf("Matrix C \n");
  display_matrix(h_c.data());

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

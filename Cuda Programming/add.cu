#include <stdio.h>
#include <stdlib.h> // Required for srand() and rand()
#include <time.h>   // Required for time()

#define N 10

int main()
{
    float randomNumbers[N];
    srand(time(NULL));
    for (int i = 0; i < N; ++i)
    {
        randomNumbers[i] = (float)rand() / RAND_MAX;
    }

    // Print the array of random numbers
    printf("Random numbers:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.6f\n", randomNumbers[i]); // Print with 6 decimal places
    }

    return 0;

    // float a[N], b[N], c[N]; // host copies of a, b, c
    // int *d_a, *d_b, *d_c;   // device copies of a, b, c
    // int size = N * sizeof(float);

    // // Allocate space for device copies of a, b, c
    // cudaMalloc((void **)&d_a, size);
    // cudaMalloc((void **)&d_b, size);
    // cudaMalloc((void **)&d_c, size);

    // // Setup input values
    // for (int i = 0; i < N; i++)
    // {
    //     a[i] = i;
    //     b[i] = i * i;
    // }

    // // Copy inputs to device
    // cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // // Launch add() kernel on GPU
    // add<<<1, N>>>(d_a, d_b, d_c);

    // // Copy result back to host
    // cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // // Print the result
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

    // // Clean up
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    // return 0;
}
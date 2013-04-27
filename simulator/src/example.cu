/*
 * This program uses the device CURAND API to calculate what
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define N 1024
#define THREADS_PER_BLOCK 24
#define BLOCKS_PER_RUN 16
#define THREADS (THREADS_PER_BLOCK*BLOCKS_PER_RUN)
#define RUNS (N/BLOCKS_PER_RUN)
#define RANDOMS_PER_ITERATION 10000

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
//printf("setup_kernel[%04d]\n", id);
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state,
                                unsigned int *result,
                                unsigned int *chunk)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
//printf("generate_kernel[(%04d * %04d) + %04d == %04d]\n", blockDim.x, blockIdx.x, threadIdx.x, id);
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int n = 0; n < RANDOMS_PER_ITERATION; n++) {
        x = curand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
printf("generate_kernel[(%04d * %04d) + %04d == %04d]\n", blockDim.x, *chunk, id, (blockDim.x * *chunk) + id);
    result[(blockDim.x * *chunk) + id] += count;
}

int main(int argc, char *argv[])
{
    unsigned int i;
    unsigned int total;
    curandState *devStates;
    unsigned int *devResults, *hostResults;

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(N, sizeof(unsigned int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, N * sizeof(unsigned int)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, N * sizeof(unsigned int)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, THREADS * sizeof(curandState)));


    // Set up RNG state objects.
    setup_kernel<<<BLOCKS_PER_RUN, THREADS_PER_BLOCK>>>(devStates);
//    cudaDeviceSynchronize();

    for(i = 0; i < RUNS; i++) {
      generate_kernel<<<BLOCKS_PER_RUN, THREADS_PER_BLOCK>>>(devStates, devResults, &i);
      cudaDeviceSynchronize();
    }


    // Copy device memory to host.
//    CUDA_CALL(cudaMemcpy(hostResults, devResults, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost));

sleep(10);

    // Show result.
    total = 0;
    for(i = 0; i < N; i++) {
        total += hostResults[i];
    }
    printf("Fraction with low bit set was %10.13f\n",
        (float)total / (1.0f * N * RANDOMS_PER_ITERATION));


    /* Cleanup */
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}

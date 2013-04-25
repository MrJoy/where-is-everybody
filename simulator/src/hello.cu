#include <stdio.h>
#include "curand_kernel.h"

const int N = 16;
const int blocksize = 16;

__global__
void hello(unsigned int *seeds, unsigned int *outs, int n)
{
  curandStateXORWOW_t rgen ;
  curand_init(seeds[threadIdx.x], 0,0,&rgen);
  for(int i=0; i<n; ++i){
    outs[threadIdx.x * n + i] = curand(&rgen) ;
  }
  //see float curand_uniform
  //see curand__discrete
}

void get_seeds(unsigned int *seeds, int n){
  FILE* randomSource = fopen("/dev/random", "rb");
  unsigned int seed0;
  int recordsRead = fread(&seed0, sizeof(unsigned long), 1, randomSource);
  //assert(recordsRead == n);
  for(int i=0; i<n; ++i){
    seeds[i] = seed0 + i;
  }
  fclose(randomSource);
}

void inspect(unsigned int *array, int n){
  for(int i=0; i< n; ++i){
    printf("%x ", array[i]);
  }
  printf("\n");
}

int main()
{
  unsigned int seeds[N];
  unsigned int outputs[N * N];

  unsigned int *cseeds;
  unsigned int *coutputs;
  const int seed_size = N*sizeof(unsigned int);
  const int output_size = N*N*sizeof(unsigned int);

  get_seeds(&seeds[0], N );
  inspect(seeds, N);

  cudaMalloc( (void**)&cseeds, seed_size );
  cudaMalloc( (void**)&coutputs, output_size );
  cudaMemcpy( cseeds, seeds, seed_size, cudaMemcpyHostToDevice );
  cudaMemcpy( coutputs, outputs, output_size, cudaMemcpyHostToDevice );

  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  //TODO, don't pass N, use dim
  hello<<<dimGrid, dimBlock>>>(cseeds, coutputs, N);
  cudaMemcpy( outputs, coutputs, output_size, cudaMemcpyDeviceToHost );
  cudaFree( cseeds );
  cudaFree( coutputs );

  for(int i=0; i<N; ++i){
    inspect(outputs + i * N, N );
  }
  return EXIT_SUCCESS;
}

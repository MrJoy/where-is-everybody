#include <stdio.h>
#include "curand_kernel.h"

const int STARS                   = 1024 * 1024 * 64 ;//2**26

const int THREADS_PER_BLOCK       = 32; //2**5
const int BLOCKS                  = 128; //2**7
const int THREADS_EVER            = THREADS_PER_BLOCK * BLOCKS ;//2**12

const int NEIGHBORHOODS           = THREADS_EVER ;
const int NEIGHBORHOOD_STARS      = STARS / NEIGHBORHOODS ;//2**14

typedef unsigned char output_t ;
const char * OUTPUT_T_FORMAT = "%x " ;

__global__
void
init_rands(unsigned int seed, curandStateXORWOW_t *rgens )
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init( seed, baseIdx, 0, & rgens[baseIdx]);
  //printf( "init_rands %i %u thr=%i blck=%i blckdim=%i\n", baseIdx, seed, threadIdx.x, blockIdx.x, blockDim.x );
}

__global__
void
generate_rands( curandStateXORWOW_t *rgens, output_t *outs, int n)
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curandStateXORWOW_t rgen = rgens[baseIdx];
  for( int i=0; i<n; ++i ) {
    outs[baseIdx * n + i] = static_cast<output_t>(
        ceil( curand_uniform( &rgen ) - 0.5 ));
    //printf( "generate_rands baseIdx=%i i=%i out=%08x\n", baseIdx, i, outs[baseIdx * n + i] );
  }
  rgens[baseIdx] = rgen;
}

unsigned int
generate_seed()
{
  FILE* randomSource = fopen("/dev/random", "rb");
  unsigned int seed;
  int recordsRead = fread( &seed, sizeof(unsigned int), 1, randomSource );
  fclose( randomSource );
  return seed;
}

void
inspect( output_t *array, int line, int n )
{
  for( int i=0; i<n; ++i ) {
    printf( OUTPUT_T_FORMAT, array[i] );
    if( (i + 1) % line == 0 ){
      printf( "\n" );
    }
  }
  printf( "\n" );
}

void
inspect_sum( output_t *array, int line, int n )
{
  for( int i=0; i<n; ++i ) {
    if( (i + 1) % line == 0 ){
      printf( OUTPUT_T_FORMAT, array[i] );
      printf( "\n" );
    }
  }
  printf( "\n" );
}

int
main()
{
  output_t *outputs ;
  output_t *coutputs;
  curandStateXORWOW_t *crgens;

  unsigned int seed = generate_seed();

  const int output_size = STARS * sizeof( output_t );
  const int rgen_size =   NEIGHBORHOODS * sizeof( curandStateXORWOW_t );

  cudaMalloc( (void**)&coutputs, output_size );
  cudaMalloc( (void**)&crgens, rgen_size );
  outputs = static_cast<output_t *>(malloc( output_size ));

  init_rands<<<BLOCKS, THREADS_PER_BLOCK>>>( seed, crgens );
  generate_rands<<<BLOCKS, THREADS_PER_BLOCK>>>( crgens, coutputs, NEIGHBORHOOD_STARS );

  cudaMemcpy( outputs, coutputs, output_size, cudaMemcpyDeviceToHost );
  cudaFree( coutputs );
  cudaFree( crgens );

  inspect_sum( outputs, NEIGHBORHOOD_STARS, STARS );
  return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"
#include "stars_config.h"
#include "stars_helpers.h"
#include "stars_helpers_c.h"

int
main()
{
  output_t *outs ;
  output_t *couts1;
  output_t *couts2;
  output_t *cstate_matrix;
  output_t *cstate_matrix_shadow;
  float *cpchange;
  curandStateXORWOW_t *crgens;

  unsigned int seed = generate_seed();

  const int output_size = STARS * sizeof( output_t );
  const int rgen_size =   NEIGHBORHOODS * sizeof( curandStateXORWOW_t );
  const int state_matrix_size = 2 * NUM_STATES * sizeof( output_t );
  const int pchange_size = NUM_STATES * sizeof( float );

  cudaMalloc( (void**)&couts1, output_size );
  cudaMalloc( (void**)&couts2, output_size );
  cudaMalloc( (void**)&cstate_matrix, state_matrix_size );
  cudaMalloc( (void**)&cpchange, pchange_size );
  cudaMalloc( (void**)&crgens, rgen_size );


  //TODO: are these being copied to the host correctly?
  cudaMemcpy( cstate_matrix, STATE_CHANGES, state_matrix_size, cudaMemcpyHostToDevice );
  cudaMemcpy( cpchange, P_CHANGE, pchange_size, cudaMemcpyHostToDevice );


  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>( couts1, PROTOSTAR, NEIGHBORHOOD_STARS );
  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>( couts2, PROTOSTAR, NEIGHBORHOOD_STARS );
  init_rands<<<BLOCKS, THREADS_PER_BLOCK>>>( seed, crgens );


  cudaHostAlloc(&cstate_matrix_shadow, state_matrix_size, sizeof(output_t));
  cudaMemcpy( cstate_matrix_shadow, cstate_matrix, state_matrix_size, cudaMemcpyDeviceToHost );
  inspect( 'S', STATE_CHANGES, 2, state_matrix_size);
  inspect( 's', cstate_matrix_shadow, 2, state_matrix_size);

  cudaHostAlloc(&outs, STARS, sizeof(output_t));
  cudaMemcpy( outs, couts1, output_size, cudaMemcpyDeviceToHost );
  inspect( 'x', outs, NEIGHBORHOOD_STARS, STARS );
  cudaMemcpy( outs, couts2, output_size, cudaMemcpyDeviceToHost );
  inspect( 'X', outs, NEIGHBORHOOD_STARS, STARS );

  for( int i=0; i< ITERATIONS; i += 2 ){
    iterate_states<<<BLOCKS, THREADS_PER_BLOCK>>>( crgens, couts1, couts2, NEIGHBORHOOD_STARS, cstate_matrix, cpchange );
    iterate_states<<<BLOCKS, THREADS_PER_BLOCK>>>( crgens, couts2, couts1, NEIGHBORHOOD_STARS, cstate_matrix, cpchange );
    //TODO switch to an explicit cuda 3d structure, then compute infections
    cudaMemcpy( outs, couts1, output_size, cudaMemcpyDeviceToHost );
    inspect( 'y', outs, NEIGHBORHOOD_STARS, STARS );
  }

  cudaFree( couts1 );
  cudaFree( couts2 );
  cudaFree( crgens );
  cudaFree( cstate_matrix );
  cudaFree( cpchange );

  return EXIT_SUCCESS;
}

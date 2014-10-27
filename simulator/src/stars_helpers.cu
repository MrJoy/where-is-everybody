#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"
#include "stars_helpers.h"

__global__ void
init_rands(unsigned int seed, curandStateXORWOW_t *rgens )
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init( seed, baseIdx, 0, & rgens[baseIdx]);
}

__global__ void
init_buf( output_t *outs, output_t value, int n)
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  for( int i=0; i<n; ++i ) {
    outs[baseIdx * n + i] = value;
  }
}

__global__ void
iterate_states(
    curandStateXORWOW_t *rgens,
    output_t *buf_in,
    output_t *buf_out,
    int neighborhood_stars,
    output_t *state_matrix,
    float *pchange
    )
{
  int neighborhood = threadIdx.x + blockIdx.x * blockDim.x;
  int base = neighborhood * neighborhood_stars;
  curandStateXORWOW_t rgen = rgens[neighborhood];
  for( int i=0; i<neighborhood_stars; ++i ) {
    const int star = base + i;
    output_t old_state = buf_in[star];
    buf_out[star] = state_matrix[
        old_state * 2 +
        (1 -
          (unsigned int) ceil(curand_uniform(&rgen) -
          pchange[ old_state ]))
      ];
  }
  rgens[neighborhood] = rgen;
}

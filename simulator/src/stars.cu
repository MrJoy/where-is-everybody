#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"
#include "stars_host_device_shared.h"
#include "stars_config.h"
#include "stars_helpers.h"
#include "stars_helpers_c.h"

// 10myr/time step gives 10 Billion year simulation
#define ITERATIONS 1000

int
main()
{
  // output_t *outs ;
  output_t *couts1;
  output_t *couts2;
  curandStateXORWOW_t *crgens;
  uint32_t *thread_star_state_counters;
  uint32_t *star_state_counters;
  uint32_t host_star_state_counters[NUM_STATES];

  unsigned int seed = generate_seed();

  const int output_size = STARS * sizeof( output_t );
  const int rgen_size =   THREADS_EVER * sizeof( curandStateXORWOW_t );

  cudaMalloc( (void**)&couts1, output_size );
  cudaMalloc( (void**)&couts2, output_size );
  cudaMalloc( (void**)&crgens, rgen_size );
  cudaMalloc( (void**)&thread_star_state_counters, NUM_STATES * THREADS_EVER * sizeof(uint32_t) );
  cudaMalloc( (void**)&star_state_counters, NUM_STATES * sizeof(uint32_t) );


  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>(couts1);
  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>(couts2);
  init_rands<<<BLOCKS, THREADS_PER_BLOCK>>>(seed, crgens);
  init_counters<<<BLOCKS, THREADS_PER_BLOCK>>>(thread_star_state_counters);
  // star_state_counters is re-initialized by each call to sum_states.

  // cudaHostAlloc(&outs, STARS, sizeof(output_t));
  cudaDeviceSynchronize();

  // for( int i=0; i< ITERATIONS; i += 2 ){
  for( int i=0; i< 100; i += 2 ){
    iterate_states<<<BLOCKS, THREADS_PER_BLOCK>>>(crgens, couts1, couts2);
    cudaDeviceSynchronize();
    iterate_states<<<BLOCKS, THREADS_PER_BLOCK>>>(crgens, couts2, couts1);
    cudaDeviceSynchronize();
    count_states<<<BLOCKS, THREADS_PER_BLOCK>>>(couts1, thread_star_state_counters);
    cudaDeviceSynchronize();
    sum_states<<<1, 1>>>(thread_star_state_counters, star_state_counters);
    cudaDeviceSynchronize();
    cudaError_t ret = cudaMemcpy(host_star_state_counters, star_state_counters, NUM_STATES * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    show_counters((int)ret, i, host_star_state_counters);
  }

  cudaFree( couts1 );
  cudaFree( couts2 );
  cudaFree( crgens );
  cudaDeviceReset();
  // show_counters(0, host_star_state_counters);

  return EXIT_SUCCESS;
}

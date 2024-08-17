#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"
#include "stars_host_device_shared.h"

#include "stdio.h"
// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif
/*
// the code I want
enum class States : output_t { 
  PROTOSTAR = 0,
  UNINHABITABLE = 1,
  INHABITABLE = 2,
  CELLULAR = 3, 
  OXYGEN_EVENT = 4,
  CAMBRIAN_EVENT = 5,
  TECHNICAL_CIV = 6
  //SPACE_FARING = 7
};
*/
// the code I have: nvcc does not support -std=c++0x.
// I am refusing to deal with pre c++11 enum.

// initial state is PROTOSTAR, set from device code.
// Our state machine can't go back to that so we don't actually
// define it here to avoid compiler warnings.
// __device__ __const__ output_t  PROTOSTAR = 0;
__device__ __const__ output_t  UNINHABITABLE = 1;
__device__ __const__ output_t  INHABITABLE = 2;
__device__ __const__ output_t  CELLULAR = 3;
__device__ __const__ output_t  OXYGEN_EVENT = 4;
__device__ __const__ output_t  CAMBRIAN_EVENT = 5;
__device__ __const__ output_t  TECHNICAL_CIV = 6;
__device__ __const__ output_t  SPACE_FARING = 7;

// **********************************************************************
// each step is a boolean trial representing 10myr.
//
// the steps are indexed by their state values.
//
// at each step we generate a flat random 0.0f to 1.0f.
//
// Then we convert to 0 if less than the transition probability or 1
// if more than transition probability.  This means that 0 is a transition
// and 1 is no change. 
//
// note civ states (eg: spacefaring, collapse, colonize) are modeled 
// separately in fast time since 10myr steps don't seem reasonable
//
// STATE_CHANGES is indexed by pseudo code
//   flip{0,1} * NUM_STATES + current_state{0..(NUM_STATES -1)}
//
//   where flip=0 implies transistion and 1 implies stasis and
//   where the value returned by that is the next state
//**********************************************************************

__device__ __const__ float pZ = 0.0 ;   //uninhabitable
__device__ __const__ float pH = 0.5 ;   //habitable
__device__ __const__ float pB = 0.007 ; //Cellular Biology (50% chance after 100*10myr )
__device__ __const__ float pO = 0.007 ; //Oxygen Event (took another 100 iterations on earth)
__device__ __const__ float pM = 0.0035; //Cambrian Explosion of [M]ulticellular life (took 200 iterations on earth)
__device__ __const__ float pT = 0.014;  //Technical Civilization (radio telescope)
__device__ __const__ float pS = 0.5  ;  //Space Faring Civilization (or collapse) per 1000 years of Tech Civ
__device__ __const__ float pC = 0.5  ;  //spacefaring local solar system civ collapse

__device__ __const__ output_t STATE_CHANGES [ NUM_STATES * 2 ] =
{
  UNINHABITABLE,  INHABITABLE,      //0,1 states from PROTOSTAR,
  UNINHABITABLE,  UNINHABITABLE,    //0,1 states from UNINHABITABLE
  INHABITABLE,    CELLULAR,         //0,1 states from INHABITABLE
  CELLULAR,       OXYGEN_EVENT,     //0,1 states from CELLULAR
  OXYGEN_EVENT,   CAMBRIAN_EVENT,   //0,1 states from OXYGEN_EVENT
  CAMBRIAN_EVENT, TECHNICAL_CIV,    //0,1 states from CAMBRIAN_EVENT
  CAMBRIAN_EVENT, SPACE_FARING,     //0,1 states from TECHNICAL_CIV
  SPACE_FARING,   TECHNICAL_CIV,    //0,1 states from SPACE_FARING
};

__device__ __const__ float P_CHANGE [ NUM_STATES ] =
{
  pH,  //p(change) from PROTOSTAR,
  pZ,  //p(change) from UNINHABITABLE
  pB,  //p(change) from INHABITABLE
  pO,  //p(change) from CELLULAR
  pM,  //p(change) from OXYGEN_EVENT
  pT,  //p(change) from CAMBRIAN_EVENT
  pS,  //p(change) from TECHNICAL_CIV
  pC,  //p(change) from SPACE_FARING
};


__device__ __const__ int NUM_STARS_PER_THREAD = STARS_PER_THREAD;
__device__ __const__ int NUM_THREADS_EVER = THREADS_EVER;


__global__ void
init_rands(unsigned int seed, curandStateXORWOW_t *rgens )
{
  const int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init( seed, baseIdx, 0, & rgens[baseIdx]);
}

__global__ void
init_buf(output_t *outs)
{
  const int count = NUM_STARS_PER_THREAD;
  const int base = (threadIdx.x + blockIdx.x * blockDim.x) * count;
  // printf("INIT_BUF ti=%i tidx=%i blkidx=%i blkdim=%i base=%i count=%i\n",
  //     threadIdx.x + blockIdx.x * blockDim.x, threadIdx.x, blockIdx.x, blockDim.x, base, count);
  for(int i=0; i < count; ++i) {
    outs[base + i] = 0;
  }
}

__global__ void
init_counters(counter_t *counters)
{
  const int count = NUM_STATES;
  const int base = (threadIdx.x + blockIdx.x * blockDim.x)*count;
  // printf("INIT_CNTRS ti=%i tidx=%i blkidx=%i blkdim=%i base=%i count=%i\n",
  //     threadIdx.x + blockIdx.x * blockDim.x, threadIdx.x, blockIdx.x, blockDim.x, base, count);
  for(int i=0; i < count; ++i) {
    counters[base + i] = 0;
  }
}

__global__ void
iterate_states(
    curandStateXORWOW_t *rgens,
    output_t *buf_in,
    output_t *buf_out)
{
  int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  int base = thread_index * NUM_STARS_PER_THREAD;
  // printf("ti=%i tidx=%i blkidx=%i blkdim=%i base=%i\n",
  //     thread_index, threadIdx.x, blockIdx.x, blockDim.x, base);
  curandStateXORWOW_t rgen = rgens[thread_index];
  for( int i=0; i<NUM_STARS_PER_THREAD; ++i ) {
    const int star = base + i;
    output_t old_state = buf_in[star];
    buf_out[star] = STATE_CHANGES[
        old_state * 2 +
        (1 -
          (unsigned int) ceil(curand_uniform(&rgen) - P_CHANGE[ old_state ])
        )
      ];
  }
  rgens[thread_index] = rgen;
}

__global__ void
count_states(output_t *buf, counter_t *counters)
{
  int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  int stars_base = thread_index * NUM_STARS_PER_THREAD;
  int counters_base = thread_index * NUM_STATES;
  counter_t local_counters[NUM_STATES] = {0};
  // printf("ti=%i tidx=%i blkidx=%i blkdim=%i stars_base=%i\n",
  //     thread_index, threadIdx.x, blockIdx.x, blockDim.x, stars_base);
  for( int i=0; i < NUM_STARS_PER_THREAD; ++i ) {
    const int star = stars_base + i;
    output_t state = buf[star];
    local_counters[state] += 1;
  }
  counters[counters_base] = local_counters[0];
  counters[counters_base + 1] = local_counters[1];
  counters[counters_base + 2] = local_counters[2];
  counters[counters_base + 3] = local_counters[3];
  counters[counters_base + 4] = local_counters[4];
  counters[counters_base + 5] = local_counters[5];
  counters[counters_base + 6] = local_counters[6];
  counters[counters_base + 7] = local_counters[7];
  // printf("ti=%i tidx=%i blkidx=%i blkdim=%i stars_base=%i counters_base=%i %u %u %u %u %u %u %u %u\n",
  //     thread_index, threadIdx.x, blockIdx.x, blockDim.x, stars_base, counters_base,
  //     local_counters[0], local_counters[1], local_counters[2], local_counters[3],
  //     local_counters[4], local_counters[5], local_counters[6], local_counters[7]
  //     );
}

__global__ void
sum_states(counter_t *counters, counter_t *sums)
{
  for( int i=0; i < NUM_STATES; ++i ) {
    sums[i] = 0;
  }
  for( int i=0; i < NUM_THREADS_EVER; ++i ) {
    for( int j=0; j < NUM_STATES; ++j ) {
      sums[j] = sums[j] + counters[i * NUM_STATES + j];
    }
  }
  // printf("NUM_THREADS_EVER=%i sums: %u %u %u %u %u %u %u %u\n",
  //     NUM_THREADS_EVER,
  //     sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6], sums[7]);
}



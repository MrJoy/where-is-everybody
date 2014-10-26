#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"


const int THREADS_PER_BLOCK       = 32; //2**5
const int BLOCKS                  = 128; //2**7
const int THREADS_EVER            = THREADS_PER_BLOCK * BLOCKS ;//2**12

//const int STARS                   = 1024 * 1024 * 64 ;//2**26
const int STARS                   = 1024 * 1024 * 4;//2**16
//const int STARS                   = THREADS_EVER * 2;
const int NEIGHBORHOODS           = THREADS_EVER ;
const int NEIGHBORHOOD_STARS      = STARS / NEIGHBORHOODS ;//2**14

const int ITERATIONS              = 1000 ;//at 1m years / iteration
// when debugging this is the minimal
// const int ITERATIONS              = 2 ;//at 1m years / iteration

typedef uint8_t output_t ;
const char * OUTPUT_T_FORMAT = "%u" ;

/*
 * see notes-20121226.md for description of the semantics of the sim
 */

/*
// the code I want
enum class States : output_t { 
  INIT = 0,
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
const output_t  INIT = 0;
const output_t  UNINHABITABLE = 1;
const output_t  INHABITABLE = 2;
const output_t  CELLULAR = 3;
const output_t  OXYGEN_EVENT = 4;
const output_t  CAMBRIAN_EVENT = 5;
const output_t  TECHNICAL_CIV = 6;

const unsigned int  NUM_STATES = 7;
 
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
// STATE_MATRIX is indexed by pseudo code
//   flip{0,1} * NUM_STATES + current_state{0..(NUM_STATES -1)}
//
//   where flip=0 implies transistion and 1 implies stasis and
//   where the value returned by that is the next state
//**********************************************************************

const output_t STATE_MATRIX [14] =
{
  //transition
  INHABITABLE,
  UNINHABITABLE,
  CELLULAR,
  OXYGEN_EVENT,
  CAMBRIAN_EVENT,
  TECHNICAL_CIV,
  TECHNICAL_CIV,
  //stasis
  UNINHABITABLE,
  UNINHABITABLE,
  INHABITABLE,
  CELLULAR,
  OXYGEN_EVENT,
  CAMBRIAN_EVENT,
  TECHNICAL_CIV
};



const float pH = 0.5 ;   //habitable
const float pB = 0.007 ; //Cellular Biology (50% chance after 100*10myr )
const float pO = 0.007 ; //Oxygen Event (took another 100 iterations on earth)
const float pM = 0.0035; //Cambrian Explosion (took 200 iterations on earth)
const float pT = 0.014;  //Technical Civilization
//const float pS = 0.5  ;  //Space Faring Civilization

const float PCHANGE[7] = { pH, 0.0, pB, pO, pM, pT, 0.0 };


__global__ void
init_rands(unsigned int seed, curandStateXORWOW_t *rgens )
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init( seed, baseIdx, 0, & rgens[baseIdx]);
  //printf( "init_rands %i %u thr=%i blck=%i blckdim=%i\n", baseIdx, seed, threadIdx.x, blockIdx.x, blockDim.x );
}

__global__ void
generate_rands( curandStateXORWOW_t *rgens, output_t *outs, int n)
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  curandStateXORWOW_t rgen = rgens[baseIdx];
  for( int i=0; i<n; ++i ) {
    outs[baseIdx * n + i] = static_cast<output_t>(
        ceil( curand_uniform( &rgen ) - 0.5 ));
  }
  rgens[baseIdx] = rgen;
}

__global__ void
init_buf( output_t *outs, int n)
{
  int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
  for( int i=0; i<n; ++i ) {
    outs[baseIdx * n + i] = INIT;
  }
}

__global__ void
iterate_states( curandStateXORWOW_t *rgens, output_t *buf_in, output_t *buf_out, int neighborhood_stars, output_t *state_matrix, float *pchange )
{
  int neighborhood = threadIdx.x + blockIdx.x * blockDim.x;
  int base = neighborhood * neighborhood_stars;
  curandStateXORWOW_t rgen = rgens[neighborhood];
  for( int i=0; i<neighborhood_stars; ++i ) {
    int star = base + i;
    output_t old_state = buf_in[star];
    unsigned int flip = (unsigned int) ceil(
        curand_uniform(&rgen) - pchange[ old_state ]);
    buf_out[star] = state_matrix[ flip * NUM_STATES + old_state ];

    //int matrix_idx = flip * NUM_STATES + old_state ;
    //printf("neighborhood=%i neighborhood_stars=%i i=%i star=%i old_state=%i new_state=%i flip=%i p=%f matrix_idx=%i\neighborhood_stars",
    //  neighborhood, neighborhood_stars, i, 
    //  star, old_state, buf_out[star], flip,
    //  pchange[ old_state], matrix_idx );
  }
  rgens[neighborhood] = rgen;
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

#define STAR_STATE_CHAR(x) (int)((output_t)('0') + x)

void
inspect( const char prefix, const output_t *array, const int line, const int n )
{
  for( int i=0; i<n; ++i ) {
    if( i % line == 0 ){
      putchar(prefix);
    }
    putchar(STAR_STATE_CHAR(array[i]));
    if( (i + 1) % line == 0 ){
      putchar('\n');
    }
  }
}

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

  cudaMemcpy( cstate_matrix, STATE_MATRIX, state_matrix_size, cudaMemcpyHostToDevice );
  cudaMemcpy( cpchange, PCHANGE, pchange_size, cudaMemcpyHostToDevice );

  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>( couts1, NEIGHBORHOOD_STARS );
  init_buf<<<BLOCKS, THREADS_PER_BLOCK>>>( couts2, NEIGHBORHOOD_STARS );
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
    //debug
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

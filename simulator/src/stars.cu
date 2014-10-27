#include <stdio.h>
#include <stdint.h>
#include "curand_kernel.h"
#include "stars_helpers.h"

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

//TODO: this is defined in stars_helpers.h but it should be in a
//different shared header:
//
//typedef uint8_t output_t ;

/*
 * see notes-20121226.md for description of the semantics of the sim
 */

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
const output_t  PROTOSTAR = 0;
const output_t  UNINHABITABLE = 1;
const output_t  INHABITABLE = 2;
const output_t  CELLULAR = 3;
const output_t  OXYGEN_EVENT = 4;
const output_t  CAMBRIAN_EVENT = 5;
const output_t  TECHNICAL_CIV = 6;
const output_t  SPACE_FARING = 7;

const unsigned int  NUM_STATES = 8;

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

const float pZ = 0.0 ;   //uninhabitable
const float pH = 0.5 ;   //habitable
const float pB = 0.007 ; //Cellular Biology (50% chance after 100*10myr )
const float pO = 0.007 ; //Oxygen Event (took another 100 iterations on earth)
const float pM = 0.0035; //Cambrian Explosion of [M]ulticellular life (took 200 iterations on earth)
const float pT = 0.014;  //Technical Civilization (radio telescope)
const float pS = 0.5  ;  //Space Faring Civilization (or collapse) per 1000 years of Tech Civ
const float pC = 0.5  ;  //spacefaring local solar system civ collapse

const output_t STATE_CHANGES [ NUM_STATES * 2 ] =
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

const float P_CHANGE [ NUM_STATES ] =
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

extern unsigned int generate_seed();
extern void inspect( const char prefix, const output_t *array, const int line, const int n );

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

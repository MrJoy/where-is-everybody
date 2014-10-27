#include <stdint.h>
#ifndef _STAR_CONFIG_H_
#define _STAR_CONFIG_H_ 1
typedef uint8_t output_t ;

extern const int THREADS_PER_BLOCK       ;//= 32; //2**5
extern const int BLOCKS                  ;//= 128; //2**7
extern const int THREADS_EVER            ;//= THREADS_PER_BLOCK * BLOCKS ;//2**12

//const int STARS                   ;//= 1024 * 1024 * 64 ;//2**26
extern const int STARS                   ;//= 1024 * 1024 * 4;//2**16
//const int STARS                   ;//= THREADS_EVER * 2;
extern const int NEIGHBORHOODS           ;//= THREADS_EVER ;
extern const int NEIGHBORHOOD_STARS      ;//= STARS / NEIGHBORHOODS ;//2**14

extern const int ITERATIONS              ;//= 1000 ;//at 1m years / iteration
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
extern const output_t  PROTOSTAR ;//= 0;
extern const output_t  UNINHABITABLE ;//= 1;
extern const output_t  INHABITABLE ;//= 2;
extern const output_t  CELLULAR ;//= 3;
extern const output_t  OXYGEN_EVENT ;//= 4;
extern const output_t  CAMBRIAN_EVENT ;//= 5;
extern const output_t  TECHNICAL_CIV ;//= 6;
extern const output_t  SPACE_FARING ;//= 7;

//extern const unsigned int  NUM_STATES ;//= 8;
//needed below, so ...
#define NUM_STATES 8

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

extern const float pZ ;//= 0.0 ;   //uninhabitable
extern const float pH ;//= 0.5 ;   //habitable
extern const float pB ;//= 0.007 ; //Cellular Biology (50% chance after 100*10myr )
extern const float pO ;//= 0.007 ; //Oxygen Event (took another 100 iterations on earth)
extern const float pM ;//= 0.0035; //Cambrian Explosion of [M]ulticellular life (took 200 iterations on earth)
extern const float pT ;//= 0.014;  //Technical Civilization (radio telescope)
extern const float pS ;//= 0.5  ;  //Space Faring Civilization (or collapse) per 1000 years of Tech Civ
extern const float pC ;//= 0.5  ;  //spacefaring local solar system civ collapse

extern const output_t STATE_CHANGES [ NUM_STATES * 2 ] ;
extern const float P_CHANGE [ NUM_STATES ] ;

#endif

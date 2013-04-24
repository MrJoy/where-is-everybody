///////////////////////////////////////////////////////////////////////////////
// CUDA and CURAND includes.
///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <curand.h>


///////////////////////////////////////////////////////////////////////////////
// System includes.
///////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>


///////////////////////////////////////////////////////////////////////////////
// Local includes.
///////////////////////////////////////////////////////////////////////////////
#include "wie/device.h"
#include "wie/random.h"


///////////////////////////////////////////////////////////////////////////////
// Constants.
///////////////////////////////////////////////////////////////////////////////
#define NUM_SAMPLES 64*1048576


///////////////////////////////////////////////////////////////////////////////
// CLI.
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) //int argc, char **argv
{
  using std::runtime_error;
  using std::string;

  const char* SEED_PARAM_NAME="--seed=";
  const int SEED_PARAM_SIZE=strlen(SEED_PARAM_NAME);

  const char* SAMPLES_PARAM_NAME="--samples=";
  const int SAMPLES_PARAM_SIZE=strlen(SAMPLES_PARAM_NAME);

  const char* BLOCKS_PARAM_NAME="--blocks=";
  const int BLOCKS_PARAM_SIZE=strlen(BLOCKS_PARAM_NAME);

  const char* SHOW_FIRST_PARAM_NAME="--first";
  const int SHOW_FIRST_PARAM_SIZE=strlen(SHOW_FIRST_PARAM_NAME);

  const char* SHOW_LAST_PARAM_NAME="--last";
  const int SHOW_LAST_PARAM_SIZE=strlen(SHOW_LAST_PARAM_NAME);

  const char* SHOW_ALL_PARAM_NAME="--all";
  const int SHOW_ALL_PARAM_SIZE=strlen(SHOW_ALL_PARAM_NAME);

  const char* QUIET_PARAM_NAME="--quiet";
  const int QUIET_PARAM_SIZE=strlen(QUIET_PARAM_NAME);

  try {
    unsigned long numSamples = NUM_SAMPLES;
    unsigned long seed=0;
    int numBlocks=1;
    bool  showFirst = false, explicitShowFirst = false,
          showLast = false, explicitShowLast = false,
          showAll = true, explicitShowAll = false,
          quiet = false;

    for(int argIdx = 1; argIdx < argc; argIdx++) {
      if(strncmp(SEED_PARAM_NAME, argv[argIdx], SEED_PARAM_SIZE) == 0) {
        char* paramValue = argv[argIdx] + SEED_PARAM_SIZE;
        sscanf(paramValue, "%ld", &seed);
      } else if(strncmp(SAMPLES_PARAM_NAME, argv[argIdx], SAMPLES_PARAM_SIZE) == 0) {
        char* paramValue = argv[argIdx] + SAMPLES_PARAM_SIZE;
        sscanf(paramValue, "%ld", &numSamples);
      } else if(strncmp(BLOCKS_PARAM_NAME, argv[argIdx], BLOCKS_PARAM_SIZE) == 0) {
        char* paramValue = argv[argIdx] + BLOCKS_PARAM_SIZE;
        sscanf(paramValue, "%d", &numBlocks);
      } else if(strncmp(SHOW_FIRST_PARAM_NAME, argv[argIdx], SHOW_FIRST_PARAM_SIZE) == 0) {
        showFirst = true;
        explicitShowFirst = true;
        if(!explicitShowAll) {
          showAll = false;
        }
      } else if(strncmp(SHOW_LAST_PARAM_NAME, argv[argIdx], SHOW_LAST_PARAM_SIZE) == 0) {
        showLast = true;
        explicitShowLast = true;
        if(!explicitShowAll) {
          showAll = false;
        }
      } else if(strncmp(SHOW_ALL_PARAM_NAME, argv[argIdx], SHOW_ALL_PARAM_SIZE) == 0) {
        showAll = true;
        explicitShowAll = true;
      } else if(strncmp(QUIET_PARAM_NAME, argv[argIdx], QUIET_PARAM_SIZE) == 0) {
        quiet = true;
        if(!explicitShowAll) {
          showAll = false;
        }
        if(!explicitShowFirst) {
          showFirst = false;
        }
        if(!explicitShowLast) {
          showLast = false;
        }
      }
    }

    // TODO: Grab device ID parameter...
    // TODO: Divide the work among available (selected) devices.

    ///////////////////////////////////////////////////////////////////////////
    // Generate random seed.
    ///////////////////////////////////////////////////////////////////////////
    if(seed == 0) {
      if(!quiet) {
        std::cerr << "Generating random seed." << std::endl;
      }
      FILE* randomSource = fopen("/dev/random", "rb");
      int recordsRead = fread(&seed, sizeof(unsigned long), 1, randomSource);
      assert(recordsRead == 1);
      fclose(randomSource);
    }
    if(!quiet) {
      std::cerr << "Using seed: " << seed << std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////
    // Instantiate relevant objects / allocate memory in host RAM.
    ///////////////////////////////////////////////////////////////////////////
    WIE::Device* device = new WIE::Device();
    WIE::Random* rng = new WIE::Random(*device, seed, numSamples);
    if(!quiet) {
      std::cerr << "Generating " << numBlocks << " block(s) of " << numSamples << " samples." << std::endl;
    }
    float *samples = new float[numSamples];


    ///////////////////////////////////////////////////////////////////////////
    // Generate random numbers, copy to host, and spit them out.
    ///////////////////////////////////////////////////////////////////////////
    for(int j = 0; j < numBlocks; j++) {
      rng->generate();
      rng->copyToHost(samples);

      // Specifically putting first/last on stderr but all on stdout so you can
      // use both at once -- redirecting all to a file, and having a
      // quick-check displayed...
      if(showFirst || showLast) {
        std::cerr << "Block #" << j;
        if(showFirst) {
          std::cerr << ", first value: " << samples[0];
        }
        if(showLast) {
          std::cerr << ", last value: " << samples[numSamples - 1];
        }
        std::cerr << std::endl;
      }

      if(showAll) {
        for(unsigned int i = 0; i < numSamples; i++) {
          // TODO: Something *MUCH* faster than this. >.<
          printf("%f\n", samples[i]);
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cleanup
    ///////////////////////////////////////////////////////////////////////////
    delete rng;
    delete device;
    delete samples;
  } catch (runtime_error &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    exit(1);
  }

  exit(EXIT_SUCCESS);
}

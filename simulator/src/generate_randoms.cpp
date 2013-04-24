///////////////////////////////////////////////////////////////////////////////
// CUDA and CURAND includes.
///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <curand.h>


///////////////////////////////////////////////////////////////////////////////
// System includes.
///////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <stdexcept>


///////////////////////////////////////////////////////////////////////////////
// Local includes.
///////////////////////////////////////////////////////////////////////////////
#include "cuda/device.h"
#include "cuda/random.h"


///////////////////////////////////////////////////////////////////////////////
// Constants.
///////////////////////////////////////////////////////////////////////////////
#define NUM_SAMPLES 16777216


///////////////////////////////////////////////////////////////////////////////
// CLI.
///////////////////////////////////////////////////////////////////////////////
int main() //int argc, char **argv
{
  using std::runtime_error;
  using std::string;

  try {
    // TODO: Grab device ID parameter...
    // TODO: Use clock or some such for random seed, unless specified on CLI...
    // TODO: Parameterize NUM_SAMPLES...
    // TODO: Divide the work among available (selected) devices.

    ///////////////////////////////////////////////////////////////////////////
    // Generate random seed.
    ///////////////////////////////////////////////////////////////////////////
    FILE* randomSource = fopen("/dev/random", "rb");
    unsigned long seed;
    assert(fread(&seed, sizeof(unsigned long), 1, randomSource) == 1);
    fclose(randomSource);

    ///////////////////////////////////////////////////////////////////////////
    // Instantiate relevant objects / allocate memory in host RAM.
    ///////////////////////////////////////////////////////////////////////////
    CUDA::Device *device = new CUDA::Device();
    CUDA::Random *wrapper = new CUDA::Random(device, seed, NUM_SAMPLES);
    float *samples = (float *)malloc(NUM_SAMPLES * sizeof(float));
    assert(samples);


    ///////////////////////////////////////////////////////////////////////////
    // Generate random numbers.
    ///////////////////////////////////////////////////////////////////////////
    wrapper->generate();


    ///////////////////////////////////////////////////////////////////////////
    // Copy random numbers to host.
    ///////////////////////////////////////////////////////////////////////////
    wrapper->copyToHost(samples);


    ///////////////////////////////////////////////////////////////////////////
    // Produce output.
    ///////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < NUM_SAMPLES; i++) {
      printf("%f\n", samples[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cleanup
    ///////////////////////////////////////////////////////////////////////////
    // cleanup(&curandGenerator, &hSamples, &dSamples);
    delete wrapper;
    delete device;
    free(samples);
  } catch (runtime_error &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    exit(1);
  }

  exit(EXIT_SUCCESS);
}

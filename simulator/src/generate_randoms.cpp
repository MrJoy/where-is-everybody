///////////////////////////////////////////////////////////////////////////////
// CUDA and CURAND includes.
///////////////////////////////////////////////////////////////////////////////
#include <curand.h>
#include <helper_cuda.h>


///////////////////////////////////////////////////////////////////////////////
// System includes.
///////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <string>
#include <sstream>
#include <stdexcept>

///////////////////////////////////////////////////////////////////////////////
// Constants.
///////////////////////////////////////////////////////////////////////////////
// TODO: Make these parameters...
#define NUM_SAMPLES 16777216
#define BATCH_SIZE 16777216

///////////////////////////////////////////////////////////////////////////////
// Supporting routines.
///////////////////////////////////////////////////////////////////////////////
// TODO: Make proper classes to manage lifecycle properly.
void cleanup(curandGenerator_t* curandGenerator, float** hSamples, float** dSamples)
{
  assert(curandGenerator);
  curandDestroyGenerator(*curandGenerator);
  *curandGenerator = NULL;

  assert(dSamples);
  if(*dSamples) {
    cudaFree(*dSamples);
    *dSamples = NULL;
  }

  assert(hSamples);
  if(*hSamples) {
    free(*hSamples);
    *hSamples = NULL;
  }

  cudaDeviceReset();
}

void assertCurandResult(curandStatus_t curandResult, std::string msg)
{
  if(curandResult != CURAND_STATUS_SUCCESS) {
    std::ostringstream s;
    s << msg << ", got: " << _cudaGetErrorEnum(curandResult);
    throw std::runtime_error(s.str());
  }
}

void assertCudaResult(cudaError_t cudaResult, std::string msg)
{
  if(cudaResult != cudaSuccess) {
    std::ostringstream s;
    s << msg << ", got: " << _cudaGetErrorEnum(cudaSuccess);
    throw std::runtime_error(s.str());
  }
}

void generateCurandBatch(curandGenerator_t curandGenerator, float* dSamples, int batchSize)
{
  // Generate random numbers
  curandStatus_t curandResult = curandGenerateUniform(curandGenerator, dSamples, batchSize);
  assertCurandResult(curandResult, "Could not generate random numbers");
}

int main(int argc, char **argv)
{
  using std::runtime_error;
  using std::string;

  try {
    ///////////////////////////////////////////////////////////////////////////
    // Scratch-variables we'll reuse a lot.
    ///////////////////////////////////////////////////////////////////////////
    curandStatus_t curandResult;
    cudaError_t    cudaResult;


    ///////////////////////////////////////////////////////////////////////////
    // Select CUDA device.
    ///////////////////////////////////////////////////////////////////////////
    findCudaDevice(argc, (const char **)argv);


    ///////////////////////////////////////////////////////////////////////////
    // Allocate sample array in host RAM.
    ///////////////////////////////////////////////////////////////////////////
    float *hSamples = (float *)malloc(BATCH_SIZE * sizeof(float));
    assert(hSamples);


    ///////////////////////////////////////////////////////////////////////////
    // Allocate sample array in device RAM.
    ///////////////////////////////////////////////////////////////////////////
    float *dSamples = NULL;
    cudaResult = cudaMalloc((void **)&dSamples, BATCH_SIZE * sizeof(float));
    assertCudaResult(cudaResult, "Could not allocate device memory");


    ///////////////////////////////////////////////////////////////////////////
    // Create the Random Number Generator.
    // See docs for curandRngType for available RNGs...
    ///////////////////////////////////////////////////////////////////////////
    curandGenerator_t curandGenerator;
    curandResult = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // Mersenne Twister
    assertCurandResult(curandResult, "Could not create pseudo-random number generator");


    ///////////////////////////////////////////////////////////////////////////
    // Setup initial parameters.
    ///////////////////////////////////////////////////////////////////////////
    unsigned long prngSeed = 12345; // TODO: Use clock or something...
    curandResult = curandSetPseudoRandomGeneratorSeed(curandGenerator, prngSeed);
    assertCurandResult(curandResult, "Could not set pseudo-random number generator seed");


    ///////////////////////////////////////////////////////////////////////////
    // Generate random numbers.
    ///////////////////////////////////////////////////////////////////////////
    generateCurandBatch(curandGenerator, dSamples, BATCH_SIZE);


    ///////////////////////////////////////////////////////////////////////////
    // Copy random numbers to host.
    ///////////////////////////////////////////////////////////////////////////
    cudaResult = cudaMemcpy(hSamples, dSamples, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    assertCudaResult(cudaResult, "Could not copy random numbers to host");


    ///////////////////////////////////////////////////////////////////////////
    // Produce output.
    ///////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < BATCH_SIZE; i++) {
      printf("%f\n", hSamples[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Cleanup
    ///////////////////////////////////////////////////////////////////////////
    cleanup(&curandGenerator, &hSamples, &dSamples);
  } catch (runtime_error &e) {
    printf("runtime error (%s)\n", e.what());
    exit(1);
  }

  exit(EXIT_SUCCESS);
}

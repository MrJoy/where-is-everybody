#include "cuda/random.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction.
///////////////////////////////////////////////////////////////////////////////
CUDA::Random::Random(CUDA::Device* pDevice, unsigned long pSeed, unsigned int pSampleCount, curandRngType pRngMethod)
  : device(pDevice),
    seed(pSeed),
    sampleCount(pSampleCount),
    rngMethod(pRngMethod)
{
  Init();
}

CUDA::Random::Random(CUDA::Device* pDevice, unsigned long pSeed, unsigned int pSampleCount)
  : device(pDevice),
    seed(pSeed),
    sampleCount(pSampleCount),
    rngMethod(CURAND_RNG_PSEUDO_XORWOW)
{
  Init();
}

void CUDA::Random::Init()
{
  assert(device);
  device->activate();
  assertResult(curandCreateGenerator(&generator, rngMethod), "Could not create random number generator");
  assertResult(curandSetPseudoRandomGeneratorSeed(generator, seed), "Could not set seed value");
  samples = NULL;
}

CUDA::Random::~Random()
{
  assert(generator);
  curandDestroyGenerator(generator);
  generator = NULL;

  assert(samples);
  cudaFree(samples);
  samples = NULL;
}


///////////////////////////////////////////////////////////////////////////////
// Handy helper methods.
///////////////////////////////////////////////////////////////////////////////
void CUDA::Random::assertResult(curandStatus_t result, std::string msg)
{
  if(result != CURAND_STATUS_SUCCESS) {
    std::ostringstream s;
    s << msg << ", got curandStatus_t: " << result;
    throw std::runtime_error(s.str());
  }
}


///////////////////////////////////////////////////////////////////////////////
// Main class logic.
///////////////////////////////////////////////////////////////////////////////
void CUDA::Random::generate()
{
  device->activate();
  if(!samples) {
    device->assertResult(cudaMalloc((void **)&samples, sampleCount * sizeof(float)), "Could not allocate device memory");
  }
  curandStatus_t result = curandGenerateUniform(generator, samples, sampleCount);
  assertResult(result, "Could not generate random numbers");
}

void CUDA::Random::copyToHost(float* buffer)
{
  device->assertResult(cudaMemcpy(buffer, samples, sampleCount * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy random numbers to host");
}

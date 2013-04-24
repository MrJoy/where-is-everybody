#include "wie/random.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction.
///////////////////////////////////////////////////////////////////////////////
WIE::Random::Random(WIE::Device& pDevice, unsigned long pSeed, unsigned int pSampleCount, curandRngType pRngMethod)
  : device(pDevice),
    seed(pSeed),
    sampleCount(pSampleCount),
    rngMethod(pRngMethod)
{
  Init();
}

WIE::Random::Random(WIE::Device& pDevice, unsigned long pSeed, unsigned int pSampleCount)
  : device(pDevice),
    seed(pSeed),
    sampleCount(pSampleCount),
    rngMethod(CURAND_RNG_PSEUDO_XORWOW)
{
  Init();
}

void WIE::Random::Init()
{
  device.activate();
  assertResult(curandCreateGenerator(&generator, rngMethod), "Could not create random number generator");
  assertResult(curandSetPseudoRandomGeneratorSeed(generator, seed), "Could not set seed value");
  samples = NULL;
}

WIE::Random::~Random()
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
void WIE::Random::assertResult(curandStatus_t result, const std::string& msg)
{
  if(result != CURAND_STATUS_SUCCESS) {
    std::ostringstream s;
    s << msg << ", got curandStatus_t: " << result;
    throw std::runtime_error(s.str());
  }
}

// void WIE::Random::save(const std::string& fname)
// {
//   FILE* out = fopen(fname, "wb");
//   assert(out);
//
//   int result = fwrite(hostSamples, sampleCount * sizeof(float), 1, out);
//   assert(result == 1)
//   fclose(out);
// }


///////////////////////////////////////////////////////////////////////////////
// Main class logic.
///////////////////////////////////////////////////////////////////////////////
void WIE::Random::generate()
{
  device.activate();
  if(!samples) {
    device.assertResult(cudaMalloc((void **)&samples, sampleCount * sizeof(float)), "Could not allocate device memory");
  }
  curandStatus_t result = curandGenerateUniform(generator, samples, sampleCount);
  assertResult(result, "Could not generate random numbers");
}

void WIE::Random::copyToHost(float* buffer)
{
  device.assertResult(cudaMemcpy(buffer, samples, sampleCount * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy random numbers to host");
}

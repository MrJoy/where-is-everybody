#ifndef WIE_RANDOM_H
#define WIE_RANDOM_H

#pragma once


///////////////////////////////////////////////////////////////////////////////
// CUDA and CURAND includes.
///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <curand.h>


///////////////////////////////////////////////////////////////////////////////
// System includes.
///////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <string>
#include <sstream>
#include <stdexcept>


///////////////////////////////////////////////////////////////////////////////
// Local includes.
///////////////////////////////////////////////////////////////////////////////
#include "wie/device.h"


///////////////////////////////////////////////////////////////////////////////
// Class definition.
///////////////////////////////////////////////////////////////////////////////
namespace WIE {
  class Random
  {
  public:
    Random(WIE::Device& pDevice, unsigned long pSeed, unsigned int pSampleCount, curandRngType pRngMethod);
    Random(WIE::Device& pDevice, unsigned long pSeed, unsigned int pSampleCount);
    ~Random();
    void generate();
    void copyToHost(float* buffer);

    void assertResult(curandStatus_t result, const std::string& msg);
//    void save(const std::string& fname);

  private:
    WIE::Device&     device;
    long              seed;
    unsigned long     sampleCount;
    float*            samples;
    curandRngType     rngMethod;
    curandGenerator_t generator;

    void Init();

    // NOT IMPLEMENTED!
    Random(Random& src);
    Random(const Random& src);
    Random(volatile Random& src);
    Random(const volatile Random& src);
  };
}
#endif

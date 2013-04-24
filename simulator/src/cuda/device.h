#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

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
// Class definition.
///////////////////////////////////////////////////////////////////////////////
namespace CUDA {
  using std::string;

  class Device
  {
  public:
    Device(int pDeviceID);
    Device();
    ~Device();

    void activate();
    void assertResult(cudaError_t result, const std::string& msg);


  private:
    cudaDeviceProp  properties;
    int             deviceID;

    void Init();
    cudaDeviceProp fetchDeviceProperties(int desiredDeviceID);
    bool deviceIsUsable(cudaDeviceProp deviceProperties);
    int getDeviceCount();
    int initRequestedDevice(int desiredDeviceID);
    int initBestDevice();
    int convertSMVersion2Cores(int major, int minor);
  };
}

#endif

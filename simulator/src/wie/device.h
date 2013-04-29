#ifndef WIE_DEVICE_H
#define WIE_DEVICE_H
#pragma once


///////////////////////////////////////////////////////////////////////////////
// CUDA includes.
///////////////////////////////////////////////////////////////////////////////
#include "hemi/hemi.h"


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
namespace WIE {
  using std::string;
  using std::endl;
  using std::ostringstream;
  using std::runtime_error;

  class Device
  {
  public:
    Device(int pDeviceID);
    Device();
    ~Device();

    void activate();
    void assertResult(cudaError_t result, const string& msg);


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

    // NOT IMPLEMENTED!
    Device(Device& src);
    Device(const Device& src);
    Device(volatile Device& src);
    Device(const volatile Device& src);
  };
}

#endif

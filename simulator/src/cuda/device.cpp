#include "cuda/device.h"

///////////////////////////////////////////////////////////////////////////////
// Local/private constant data.
///////////////////////////////////////////////////////////////////////////////
typedef struct {
  int version; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
  int cores;
} VersionCorePair;

int maxArchIndex = 7;
VersionCorePair coresPerVersion[] = {
  { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
  { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
  { 0x35, 192}  // Kepler Generation (SM 3.5) GK11x class
};


///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction.
///////////////////////////////////////////////////////////////////////////////
CUDA::Device::Device(int pDeviceID)
  : deviceID(pDeviceID)
{
  Init();
}

CUDA::Device::Device()
  : deviceID(-1)
{
  Init();
}

void CUDA::Device::Init()
{
  if(deviceID >= 0) {
    deviceID = initRequestedDevice(deviceID);
  } else {
    deviceID = initBestDevice();
  }
  assert(deviceID >= 0);

  properties = fetchDeviceProperties(deviceID);
  bool isUsable = deviceIsUsable(properties);
  assert(isUsable);

  activate();
}

CUDA::Device::~Device()
{
  activate();
  cudaDeviceReset();
}

int CUDA::Device::initRequestedDevice(int desiredDeviceID)
{
  int deviceCount = getDeviceCount();

  if(deviceCount == 0) {
    throw std::runtime_error("No devices supporting CUDA were found.");
  }

  if(desiredDeviceID < 0) {
    desiredDeviceID = 0;
  }

  if(desiredDeviceID > deviceCount - 1) {
    throw std::runtime_error("Requested device ID is not valid.");
  }

  return desiredDeviceID;
}

// This function returns the best GPU (with maximum GFLOPS)
int CUDA::Device::initBestDevice()
{
  // int sm_per_multiproc  = 0;
  // int max_compute_perf   = 0, max_perf_device   = 0;

  int deviceCount = getDeviceCount();

  // Find the best major SM Architecture GPU device...
  int currentDevice = 0, bestSMArch = -1;
  cudaDeviceProp deviceProperties;
  while(currentDevice < deviceCount) {
    deviceProperties = fetchDeviceProperties(currentDevice);

    if(deviceIsUsable(deviceProperties) && deviceProperties.major > bestSMArch && deviceProperties.major < 9999) {
      bestSMArch = deviceProperties.major;
    }

    currentDevice++;
  }

  // Find the best CUDA capable GPU device...
  currentDevice = 0;
  int maxComputePerf = -1, maxPerfDevice = -1, smPerMultiproc = -1;
  while(currentDevice < deviceCount) {
    deviceProperties = fetchDeviceProperties(currentDevice);

    if(deviceIsUsable(deviceProperties)) {
      if(deviceProperties.major == 9999 && deviceProperties.minor == 9999) {
        smPerMultiproc = 1;
      } else {
        smPerMultiproc = convertSMVersion2Cores(deviceProperties.major, deviceProperties.minor);
      }

      int computePerf  = deviceProperties.multiProcessorCount * smPerMultiproc * deviceProperties.clockRate;

      if(computePerf  > maxComputePerf) {
        // If we find GPU with SM major > 2, search only these...
        if(bestSMArch > 2) {
          // If our device==bestSMArch, choose this, or else pass...
          if(deviceProperties.major == bestSMArch) {
            maxComputePerf  = computePerf;
            maxPerfDevice   = currentDevice;
          }
        } else {
          maxComputePerf  = computePerf;
          maxPerfDevice   = currentDevice;
        }
      }
    }

    currentDevice++;
  }

  return maxPerfDevice;
}

int CUDA::Device::convertSMVersion2Cores(int major, int minor)
{
  int requestedVersion = (major << 4) + minor;

  int index = 0;
  while(index <= maxArchIndex) {
    if(coresPerVersion[index].version == requestedVersion) {
      return coresPerVersion[index].cores;
    }

    index++;
  }

  return coresPerVersion[maxArchIndex].cores;
}


///////////////////////////////////////////////////////////////////////////////
// Handy helper methods.
///////////////////////////////////////////////////////////////////////////////
void CUDA::Device::assertResult(cudaError_t result, const std::string& msg)
{
  if(result != cudaSuccess) {
    std::ostringstream s;
    s << msg << ", got cudaError_t [deviceID==" << deviceID << "]: " << result;
    throw std::runtime_error(s.str());
  }
}

cudaDeviceProp CUDA::Device::fetchDeviceProperties(int desiredDeviceID)
{
  cudaDeviceProp deviceProperties;
  assertResult(cudaGetDeviceProperties(&deviceProperties, desiredDeviceID), "Could not get device properties");
  return deviceProperties;
}

bool CUDA::Device::deviceIsUsable(cudaDeviceProp deviceProperties)
{
  return (deviceProperties.computeMode != cudaComputeModeProhibited) &&
    (deviceProperties.major >= 1);
}

int CUDA::Device::getDeviceCount()
{
  int deviceCount;
  assertResult(cudaGetDeviceCount(&deviceCount), "Couldn't get device count");
  return deviceCount;
}


///////////////////////////////////////////////////////////////////////////////
// Main class logic.
///////////////////////////////////////////////////////////////////////////////
void CUDA::Device::activate()
{
  assertResult(cudaSetDevice(deviceID), "Couldn't activate device");
}

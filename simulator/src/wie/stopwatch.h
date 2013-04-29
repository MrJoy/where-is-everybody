#ifndef WIE_STOPWATCH_H
#define WIE_STOPWATCH_H
#pragma once

#include "hemi/hemi.h"

#include <string>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#ifndef HEMI_CUDA_DISABLE
  #include <ctime>
#endif

#include "wie/device.h"

namespace WIE {
  using std::string;
  using std::ostream;
  using std::fixed;
  using std::setprecision;

  // Stream #0 is special, in that all other streams are sync'ed to it and
  // vice-versa.  I.E. an event in stream #0 becomes a boundary serializing
  // events in other streams.  Makes it very handy for timing (and useless)
  // for parallelization of course.  Basically though, no matter what we're
  // doing WRT streams elsewhere, we MUST be using it for timing.
  const cudaStream_t TimingStream = 0;

  class Stopwatch {
  public:
    Stopwatch(WIE::Device& pDevice);
    ~Stopwatch();
    float lastTime();
    float totalTime();
    float averageTime();
    int iterations();

    inline void start()
    {
      ensureNotRunning("Cannot begin benchmark.");

      isRunning = true;
      device.activate();
#ifndef HEMI_CUDA_DISABLE
      cudaEventRecord(stopwatchStart, TimingStream);
#else
      startTime = clock();
#endif
    }

    inline void stop()
    {
      ensureRunning("Cannot end benchmark.");

      device.activate();
#ifndef HEMI_CUDA_DISABLE
      cudaEventRecord(stopwatchStop, TimingStream);
      cudaEventSynchronize(stopwatchStop);
      cudaEventElapsedTime(&lastTimeVal, stopwatchStart, stopwatchStop);
#else
      clock_t endTime = clock();
      lastTimeVal = 1000.0 * ((float)(endTime - startTime))/CLOCKS_PER_SEC;
#endif
      totalTimeVal += lastTimeVal;
      iterationsVal++;

      isRunning = false;
      hasRun = true;
    }

    void reportLast(ostream& target);
    void reportAverage(ostream& target);


  private:
    void ensureRunning(const string& msg);
    void ensureNotRunning(const string& msg);
    void ensureHasRun(const string& msg);

    bool         isRunning, hasRun;
    WIE::Device& device;
    float        lastTimeVal, totalTimeVal;
    int          iterationsVal;
#ifndef HEMI_CUDA_DISABLE
    cudaEvent_t  stopwatchStart, stopwatchStop;
#else
    clock_t      startTime;
#endif
  };
}

#endif

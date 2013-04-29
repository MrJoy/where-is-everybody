#include "wie/stopwatch.h"

WIE::Stopwatch::Stopwatch(WIE::Device& pDevice) :
  isRunning(false),
  hasRun(false),
  device(pDevice),
  lastTimeVal(0),
  totalTimeVal(0),
  iterationsVal(0)
{
  device.activate();
#ifndef HEMI_CUDA_DISABLE
  cudaEventCreate(&stopwatchStart);
  cudaEventCreateWithFlags(&stopwatchStop, cudaEventBlockingSync);
#endif
}

WIE::Stopwatch::~Stopwatch()
{
  device.activate();
#ifndef HEMI_CUDA_DISABLE
  cudaEventDestroy(stopwatchStart);
  cudaEventDestroy(stopwatchStop);
#endif
}

float
WIE::Stopwatch::lastTime()
{
  ensureHasRun("Cannot retrieve elapsed time.");
  return lastTimeVal;
}

float
WIE::Stopwatch::totalTime()
{
  return totalTimeVal;
}

float
WIE::Stopwatch::averageTime()
{
  ensureHasRun("Cannot retrieve elapsed time.");
  return totalTimeVal/iterationsVal;
}

int
WIE::Stopwatch::iterations()
{
  return iterationsVal;
}

void
WIE::Stopwatch::reportLast(ostream& target)
{
  ensureHasRun("Cannot report elapsed time.");
  target << "Elapsed time: " << fixed << setprecision(5) << lastTime() << endl;
}

void
WIE::Stopwatch::reportAverage(ostream& target)
{
  ensureHasRun("Cannot report elapsed time.");
  target << "Average time: " << fixed << setprecision(5) << averageTime() << endl;
}


void
WIE::Stopwatch::ensureRunning(const string& msg)
{
  if( !isRunning ) {
    ostringstream s;
    s << "Benchmark is not running: " + msg;
    throw runtime_error(s.str());
  }
}

void
WIE::Stopwatch::ensureNotRunning(const string& msg)
{
  if( isRunning ) {
    ostringstream s;
    s << "Benchmark is already running: " + msg;
    throw runtime_error(s.str());
  }
}

void
WIE::Stopwatch::ensureHasRun(const string& msg)
{
  if( !hasRun ) {
    ostringstream s;
    s << "Benchmark hasn't been run yet: " + msg;
    throw runtime_error(s.str());
  }
}

#include <stdint.h>
#include <cstdio>
#include <algorithm>

#include "hemi/hemi.h"
#include "hemi/array.h"

#include "wie/device.h"
#include "wie/stopwatch.h"

#define N 64*1024

HEMI_DEV_CALLABLE_INLINE
void initElement(uint32_t* data, int idx)
{
  if ( idx < N ) {
    data[idx] = idx;
  } else {
    printf( "WTF?! %d\n", idx );
  }
}

HEMI_KERNEL(init)(uint32_t* data)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();

  for( int idx = offset; idx < N; idx += stride ) {
    initElement( data, idx );
  }
}

HEMI_DEV_CALLABLE_INLINE
void benchmarkElement(const uint32_t* inputs, uint32_t* outputs, int idx)
{
  if( idx < N ) {
    outputs[idx] = inputs[idx] * 2;
  } else {
    printf("WTF?! %d\n", idx);
  }
}

HEMI_KERNEL(benchmark)(const uint32_t* inputs, uint32_t* outputs)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();

  for( int idx = offset; idx < N; idx += stride ) {
    benchmarkElement( inputs, outputs, idx );
  }
}

int
main()
{
  using std::min;
  using std::cout;
  using std::endl;
  using std::setprecision;

  const int blockDim = 128;
  const int gridDim = min<int>(1024, (N + blockDim - 1) / blockDim);
  cout << "blockDim=" << blockDim << ", gridDim=" << gridDim << endl;

  WIE::Device device(0);
  device.activate();

  cout << "Preparing input buffer..." << endl;
  hemi::Array<uint32_t> inputs(N, false);
  cout << "Preparing output buffer..." << endl;
  hemi::Array<uint32_t> outputs(N, false);

  cout << "Initializing input buffer..." << endl;
  HEMI_KERNEL_LAUNCH(init, gridDim, blockDim, 0, 0, inputs.writeOnlyPtr());

  cout << "Running main kernel..." << endl;
  HEMI_KERNEL_LAUNCH(benchmark, gridDim, blockDim, 0, 0, inputs.readOnlyPtr(), outputs.writeOnlyPtr());

  cout << "Checking results..." << endl;
  const uint32_t* outputs_h = outputs.readOnlyHostPtr();
  for(int j = 0; j < N; j++) {
    assert(outputs_h[j] == j * 2);
  }

  return EXIT_SUCCESS;
}

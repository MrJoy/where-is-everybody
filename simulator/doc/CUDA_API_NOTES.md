void CUDART_CB MyCallback(void *data){
    printf("Inside callback %d\n", (int)data);
}
...
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0);
}





cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// They are destroyed this way:

cudaEventDestroy(start);
cudaEventDestroy(stop);
// Elapsed Time
// The events created in Creation and Destruction can be used to time the code sample of Creation and Destruction the following way:

cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>
               (outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);




cudaSetDeviceFlags



cudaSetDevice(0);               // Set device 0 as current
cudaStream_t s0;
cudaStreamCreate(&s0);          // Create stream s0 on device 0
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0
cudaSetDevice(1);               // Set device 1 as current
cudaStream_t s1;
cudaStreamCreate(&s1);          // Create stream s1 on device 1
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1

// This kernel launch will fail:
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0







cudaSurfaceBoundaryMode
  cudaBoundaryModeZero   Zero boundary mode
  cudaBoundaryModeClamp    Clamp boundary mode
  cudaBoundaryModeTrap   Trap boundary mode

cudaSurfaceFormatMode
  cudaFormatModeForced   Forced format mode
  cudaFormatModeAuto   Auto format mode

cudaTextureAddressMode
  cudaAddressModeWrap    Wrapping address mode
  cudaAddressModeClamp   Clamp to edge address mode
  cudaAddressModeMirror    Mirror address mode
  cudaAddressModeBorder    Border address mode

cudaTextureFilterMode
  cudaFilterModePoint    Point filter mode
  cudaFilterModeLinear   Linear filter mode

cudaTextureReadMode
  cudaReadModeElementType    Read texture as specified element type
  cudaReadModeNormalizedFloat    Read texture as normalized float

cudaMalloc3DArray
  cudaArrayCubemap
  cudaArrayLayered
  cudaArraySurfaceLoadStore
    Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array
  cudaArrayTextureGather
    Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array

cudaHostAllocMapped
  Map allocation into device space

cudaHostAllocPortable
  Pinned memory accessible by all CUDA contexts

cudaHostAllocWriteCombined
  Write-combined memory

cudaHostRegisterMapped
  Map registered memory into device space

cudaHostRegisterPortable
  Pinned memory accessible by all CUDA contexts

cudaIpcMemLazyEnablePeerAccess
  Automatically enable peer access between remote devices as needed

cudaLimit
  cudaLimitStackSize   GPU thread stack size
  cudaLimitPrintfFifoSize    GPU printf/fprintf FIFO size
  cudaLimitMallocHeapSize    GPU malloc heap size

cudaMemcpyKind
  cudaMemcpyHostToHost   Host -> Host
  cudaMemcpyHostToDevice   Host -> Device
  cudaMemcpyDeviceToHost   Device -> Host
  cudaMemcpyDeviceToDevice   Device -> Device
  cudaMemcpyDefault    Default based unified virtual address space

cudaComputeMode
  cudaComputeModeDefault   Default compute mode (Multiple threads can use cudaSetDevice() with this device)
  cudaComputeModeExclusiveProcess    Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)

cudaDeviceLmemResizeToMax
  Device flag - Keep local memory allocation after launch

cudaDeviceMapHost
  Device flag - Support mapped pinned allocations

cudaDevicePropDontCare
  Empty device properties


cudaDeviceScheduleBlockingSync
  Device flag - Use blocking synchronization

cudaDeviceScheduleSpin
  Device flag - Spin default scheduling

cudaDeviceScheduleYield
  Device flag - Yield default scheduling

cudaEventBlockingSync

cudaFuncCache
  cudaFuncCachePreferNone    Default function cache configuration, no preference
  cudaFuncCachePreferShared    Prefer larger shared memory and smaller L1 cache
  cudaFuncCachePreferL1    Prefer larger L1 cache and smaller shared memory
  cudaFuncCachePreferEqual   Prefer equal size L1 cache and shared memory

cudaGraphicsMapFlags
  cudaGraphicsMapFlagsNone   Default; Assume resource can be read/written
  cudaGraphicsMapFlagsReadOnly   CUDA will not write to this resource
  cudaGraphicsMapFlagsWriteDiscard   CUDA will only write to and will not read from this resource

cudaGraphicsRegisterFlags
  cudaGraphicsRegisterFlagsNone    Default
  cudaGraphicsRegisterFlagsReadOnly    CUDA will not write to this resource
  cudaGraphicsRegisterFlagsWriteDiscard    CUDA will only write to and will not read from this resource
  cudaGraphicsRegisterFlagsSurfaceLoadStore    CUDA will bind this resource to a surface reference
  cudaGraphicsRegisterFlagsTextureGather   CUDA will perform texture gather operations on this resource


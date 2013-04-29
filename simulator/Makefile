###############################################################################
# OS / platform details.
###############################################################################
OS_KIND = $(shell uname -s 2>/dev/null)
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")
DARWIN  = $(strip $(findstring Darwin, $(OS_KIND)))


###############################################################################
# Location of compiler toolchain, CUDA/etc binaries, libraries, and shared
# sample code.
###############################################################################
ifneq ($(DARWIN),)
  CUDA_PATH       ?= $(shell find /Developer/NVIDIA -name "CUDA-*" -type d | tail -1)
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib
  GCC             ?= llvm-g++
else
  CUDA_PATH       ?= /usr/local/cuda
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64
  endif
  GCC             ?= g++
endif
CUDA_INC_PATH     ?= $(CUDA_PATH)/include
CUDA_BIN_PATH     ?= $(CUDA_PATH)/bin
CUDA_SAMPLES_PATH ?= $(CUDA_PATH)/samples/common/inc
NVCC              ?= $(CUDA_BIN_PATH)/nvcc


###############################################################################
# Build paths.
###############################################################################
SOURCE_PATH       ?= src
INTERMEDIATE_PATH ?= tmp
BUILD_PATH        ?= bin
TESTS_PATH        ?= tests


###############################################################################
# OS/architecture-specific build flags.
###############################################################################
ifneq ($(DARWIN),)
  LDFLAGS     += -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH)
  CCFLAGS     += -arch $(OS_ARCH)
  NVFLAGS     += -m64
else
  LDFLAGS     += -Xlinker -rpath $(CUDA_LIB_PATH)
  CCFLAGS     += -L$(CUDA_LIB_PATH)
  ifeq ($(OS_SIZE),32)
    CCFLAGS   += -m32
    NVFLAGS   += -m32
  else
    CCFLAGS   += -m64
    NVFLAGS   += -m64
  endif
endif
NVFLAGS       += -arch=sm_20


###############################################################################
# Debug/Optimization/Warning build flags.
###############################################################################
CCOPTFLAGS     ?=
NVOPTFLAGS     ?=
CCFLAGS        += -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
# Always include DEBUG for nvcc so that errors caught by Hemi are always
# reported.
NVFLAGS        += -DDEBUG
# Need to suss out how to ignore system headers with llvm-g++, etc...
# ifneq ($(DARWIN),)
#   CCFLAGS      += -pedantic
# endif
ifeq ($(DEBUG),1)
  CCFLAGS      += -g
  NVFLAGS      += -g -G
else
  NVOPTFLAGS +=
  ifneq ($(DARWIN),)
    CCOPTFLAGS += -O4
  else
    CCOPTFLAGS += -Os
  endif
endif


###############################################################################
# Common includes and paths for CUDA, and CURAND.
###############################################################################
CCINCLUDES := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
NVINCLUDES := -I $(CUDA_INC_PATH) -I $(SOURCE_PATH) -I $(SOURCE_PATH)/vendor
LDFLAGS    += -lcudart -lcurand


###############################################################################
# Target rules.
###############################################################################
all: $(BUILD_PATH)/randoms $(BUILD_PATH)/hello $(BUILD_PATH)/stars

$(INTERMEDIATE_PATH)/hello.o: $(SOURCE_PATH)/hello.cu
	$(NVCC) $(NVFLAGS)$(NVOPTFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/hello: $(INTERMEDIATE_PATH)/hello.o
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) -o $@ $+ $(LDFLAGS)

$(INTERMEDIATE_PATH)/example.o: $(SOURCE_PATH)/example.cu
	$(NVCC) $(NVFLAGS)$(NVOPTFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/example: $(INTERMEDIATE_PATH)/example.o
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) -o $@ $+ $(LDFLAGS)

$(INTERMEDIATE_PATH)/stars.o: $(SOURCE_PATH)/stars.cu
	$(NVCC) $(NVFLAGS)$(NVOPTFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/stars: $(INTERMEDIATE_PATH)/stars.o
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) -o $@ $+ $(LDFLAGS)

$(INTERMEDIATE_PATH)/wie/:
	mkdir -p $(INTERMEDIATE_PATH)/wie

$(INTERMEDIATE_PATH)/wie/device.o: $(SOURCE_PATH)/wie/device.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/wie/stopwatch.o: $(SOURCE_PATH)/wie/stopwatch.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/wie/random.o: $(SOURCE_PATH)/wie/random.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/randoms.o: $(SOURCE_PATH)/randoms.cpp
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) $(CCINCLUDES) -o $@ -c $<

$(BUILD_PATH)/randoms: $(INTERMEDIATE_PATH)/randoms.o $(INTERMEDIATE_PATH)/wie/random.o $(INTERMEDIATE_PATH)/wie/device.o
	$(GCC) $(CCFLAGS)$(CCOPTFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -rf tmp/*

clobber:
	rm -rf bin/*

# TODO: Better test harness, and unified control of flow via return code.
test:
	find $(TESTS_PATH) -type f -exec {} \;

full_test:
	make THOROUGH=1 test

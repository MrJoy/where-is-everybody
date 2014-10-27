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
# Determine whether or not we're actually going to generate CUDA code.
###############################################################################
SKIP_CUDA ?= 0
# TODO: Make a sane determination of whether this machine can run CUDA code and
# TODO: use that for a default...


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
# Default *FLAGS_NC to *FLAGS values inherited from environment, then accrue
# options normally...
LDFLAGS_NC     := $(LDFLAGS)
CCFLAGS_NC     := $(CCFLAGS)
NVFLAGS_NC     := $(NVFLAGS)
ifneq ($(DARWIN),)
  LDFLAGS      += -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH)
  CCFLAGS      += -arch $(OS_ARCH)
  CCFLAGS_NC   += -arch $(OS_ARCH)
  NVFLAGS      += -m64
  NVFLAGS_NC   += -arch $(OS_ARCH)
else
  LDFLAGS      += -Xlinker -rpath $(CUDA_LIB_PATH)
  CCFLAGS      += -L$(CUDA_LIB_PATH)
  ifeq ($(OS_SIZE),32)
    CCFLAGS    += -m32
    CCFLAGS_NC += -m32
    NVFLAGS    += -m32
    NCFLAGS_NC += -m32
  else
    CCFLAGS    += -m64
    CCFLAGS_NC += -m64
    NVFLAGS    += -m64
    NCFLAGS_NC += -m64
  endif
endif
# Limiting architecture to sm_20 because our main target is EC2's cg1.4xlarge
# which has a pair of Fermi M2050s.
NVFLAGS       += -arch=sm_20


###############################################################################
# Debug/Optimization/Warning build flags.
###############################################################################
CCOPTFLAGS        ?=
NVOPTFLAGS        ?=
CCFLAGS           += -DDEBUG -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
CCFLAGS_NC        += -DDEBUG -DHEMI_CUDA_DISABLE -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
# Always include DEBUG for nvcc so that errors caught by Hemi are always
# reported.
NVFLAGS           += -DDEBUG
NVFLAGS_NC        += -x c++ -DDEBUG -DHEMI_CUDA_DISABLE

# Need to suss out how to ignore system headers with llvm-g++, etc...
# ifneq ($(DARWIN),)
#   CCFLAGS         += -pedantic
# endif
ifeq ($(DEBUG),1)
  CCFLAGS         += -g
  CCFLAGS_NC      += -g
  NVFLAGS         += -g -G
  NVFLAGS_NC      += -g -G
else
  NVOPTFLAGS      += -O2
  ifneq ($(DARWIN),)
    CCOPTFLAGS    += -O4
    CCOPTFLAGS_NC += -O4
    NVOPTFLAGS_NC += -O4
  else
    CCOPTFLAGS    += -Os
    CCOPTFLAGS_NC += -Os
    NVOPTFLAGS_NC += -Os
  endif
endif


###############################################################################
# Common includes and paths for CUDA, and CURAND.
###############################################################################
CCINCLUDES    := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
CCINCLUDES_NC := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
NVINCLUDES    := -I $(CUDA_INC_PATH) -I $(SOURCE_PATH) -I $(SOURCE_PATH)/vendor
NVINCLUDES_NC := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
LDFLAGS       += -lcudart -lcurand


###############################################################################
# Allow Disabling CUDA.
###############################################################################
ifneq ($(SKIP_CUDA),0)
  NVCC       := $(GCC)
  CCFLAGS    := $(CCFLAGS_NC)
  NVFLAGS    := $(NVFLAGS_NC)
  CCOPTFLAGS := $(CCOPTFLAGS_NC)
  NVOPTFLAGS := $(NVOPTFLAGS_NC)
  CCINCLUDES := $(CCINCLUDES_NC)
  NVINCLUDES := $(NVINCLUDES_NC)
  LDFLAGS    := $(LDFLAGS_NC)
endif


###############################################################################
# Apply optimization settings last...
###############################################################################
ifneq ($(CCOPTFLAGS),)
  CCFLAGS := $(CCFLAGS) $(CCOPTFLAGS)
endif

ifneq ($(NVOPTFLAGS),)
  NVFLAGS := $(NVFLAGS) $(NVOPTFLAGS)
endif


###############################################################################
# Target rules.
###############################################################################

######################################
# Test Cases
######################################
$(INTERMEDIATE_PATH)/test/:
	mkdir -p $(INTERMEDIATE_PATH)/test
	mkdir -p $(BUILD_PATH)/test

$(INTERMEDIATE_PATH)/test/hemi.o: $(SOURCE_PATH)/test/hemi.cu $(INTERMEDIATE_PATH)/test/
	$(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/test/hemi: $(INTERMEDIATE_PATH)/test/hemi.o $(INTERMEDIATE_PATH)/wie/device.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

$(INTERMEDIATE_PATH)/test/randoms.o: $(SOURCE_PATH)/test/randoms.cpp $(INTERMEDIATE_PATH)/test/
	$(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

$(BUILD_PATH)/test/randoms: $(INTERMEDIATE_PATH)/test/randoms.o $(INTERMEDIATE_PATH)/wie/random.o $(INTERMEDIATE_PATH)/wie/device.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


######################################
# Playground
######################################
$(INTERMEDIATE_PATH)/etc/:
	mkdir -p $(INTERMEDIATE_PATH)/etc
	mkdir -p $(BUILD_PATH)/etc

$(INTERMEDIATE_PATH)/etc/hello.o: $(SOURCE_PATH)/etc/hello.cu $(INTERMEDIATE_PATH)/etc/
	$(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/etc/hello: $(INTERMEDIATE_PATH)/etc/hello.o $(INTERMEDIATE_PATH)/wie/device.o $(INTERMEDIATE_PATH)/wie/stopwatch.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

$(INTERMEDIATE_PATH)/etc/example.o: $(SOURCE_PATH)/etc/example.cu $(INTERMEDIATE_PATH)/etc/
	$(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

$(BUILD_PATH)/etc/example: $(INTERMEDIATE_PATH)/etc/example.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


######################################
# Main Code
######################################
$(INTERMEDIATE_PATH)/stars_helpers_c.o: $(SOURCE_PATH)/stars_helpers_c.cpp
	$(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/stars_helpers.o: $(SOURCE_PATH)/stars_helpers.cu
	$(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $^

$(INTERMEDIATE_PATH)/stars.o: $(SOURCE_PATH)/stars.cu
	$(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $^

$(BUILD_PATH)/stars: $(INTERMEDIATE_PATH)/stars_helpers_c.o $(INTERMEDIATE_PATH)/stars_helpers.o $(INTERMEDIATE_PATH)/stars.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


######################################
# Support Code
######################################
$(INTERMEDIATE_PATH)/wie/:
	mkdir -p $(INTERMEDIATE_PATH)/wie

$(INTERMEDIATE_PATH)/wie/device.o: $(SOURCE_PATH)/wie/device.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/wie/stopwatch.o: $(SOURCE_PATH)/wie/stopwatch.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

$(INTERMEDIATE_PATH)/wie/random.o: $(SOURCE_PATH)/wie/random.cpp $(INTERMEDIATE_PATH)/wie/
	$(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<


######################################
# Build Actions
######################################
all: $(BUILD_PATH)/stars $(BUILD_PATH)/test/randoms $(BUILD_PATH)/test/hemi $(BUILD_PATH)/etc/hello $(BUILD_PATH)/etc/example

clean:
	rm -rf tmp/*

clobber:
	rm -rf bin/*

# TODO: Better test harness, and unified control of flow via return code.
test:
	find $(TESTS_PATH) -type f -exec {} \;

full_test:
	make THOROUGH=1 test

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


###############################################################################
# Build paths.
###############################################################################
SOURCE_PATH       := src
INTERMEDIATE_PATH := tmp
BUILD_PATH        := bin


###############################################################################
# Extra user flags.
###############################################################################
EXTRA_LDFLAGS ?=


###############################################################################
# OS-specific build flags.
###############################################################################
ifneq ($(DARWIN),)
  LDFLAGS     := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
  CCFLAGS     := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
    LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
    CCFLAGS   := -m32
  else
    LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
    CCFLAGS   := -m64
  endif
endif


###############################################################################
# Debug/Optimization/Warning build flags.
###############################################################################
OPTFLAGS   :=
CCFLAGS    += -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
# Need to suss out how to ignore system headers with llvm-g++, etc...
# ifneq ($(DARWIN),)
#   CCFLAGS  += -pedantic
# endif
ifeq ($(DEBUG),1)
  CCFLAGS  += -g
else
  OPTFLAGS += -Os
endif


###############################################################################
# Common includes and paths for CUDA, and CURAND.
###############################################################################
INCLUDES := -I$(CUDA_INC_PATH) -I$(CUDA_SAMPLES_PATH)
LDFLAGS  += -lcurand


###############################################################################
# Target rules.
###############################################################################
all: $(BUILD_PATH)/generate_randoms

$(INTERMEDIATE_PATH)/generate_randoms.o: $(SOURCE_PATH)/generate_randoms.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

$(BUILD_PATH)/generate_randoms: $(INTERMEDIATE_PATH)/generate_randoms.o
	$(GCC) $(CCFLAGS)$(OPTFLAGS) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f tmp/*

clobber:
	rm -f bin/*

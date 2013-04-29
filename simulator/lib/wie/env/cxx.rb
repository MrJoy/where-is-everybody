module WIE
  module Env
    module CXX
      DEFAULT_BINS={
        #'Darwin' => 'g++-4.2',
        '_' => 'g++',
      }

      COMMON_FLAGS={
        'Darwin' => proc { "-arch #{WIE::Env::OS.arch}" },
        '_'      => proc { "-m#{WIE::Env::OS.size}" },
      }

      CODEGEN_FLAGS={
        true  => { '_' => '-g' },
        false => { '_' => '-O4', 'Darwin' => '-Os' },
      }

      COMMON_LIBS={
        '_' => proc { WIE::Env::CUDA.lib_path },
      }

# ifeq ($(DEBUG),1)
#   CCFLAGS         += -g
#   CCFLAGS_NC      += -g
#   NVFLAGS         += -g -G
#   NVFLAGS_NC      += -g -G
# else
#   NVOPTFLAGS      += -O2
#   ifneq ($(DARWIN),)
#     CCOPTFLAGS    += -O4
#     CCOPTFLAGS_NC += -O4
#     NVOPTFLAGS_NC += -O4
#   else
#     CCOPTFLAGS    += -Os
#     CCOPTFLAGS_NC += -Os
#     NVOPTFLAGS_NC += -Os
#   endif
# endif
      def self.bin
        @bin ||= [
          ENV['CC'],
          DEFAULT_BINS[WIE::Env::OS.kind]
        ].map(&:which).extant.best
      end

      def self.bin?; !@bin.blank?; end

      def self.lib_paths
        @lib_paths ||= begin
          paths = (ENV['CXX_LIB_PATHS'] || '').split(/:/).map(&:strip)
          paths += [COMMON_LIBS.best(WIE::Env::OS.kind).reify]
          paths.flatten.extant
        end
      end

      def self.common_flags
        @common_flags ||= begin
          flags = [ENV['CXX_COMMON_FLAGS']]
          flags << COMMON_FLAGS.best(WIE::Env::OS.kind).reify
          flags.flatten.clean.join(' ')
        end
      end

      def self.codegen_flags
        @codegen_flags ||= begin
          flags = [ENV['CXX_CODEGEN_FLAGS']]
          flags << CODEGEN_FLAGS[WIE::Env::Mode.debug?].best(WIE::Env::OS.kind).reify
          flags.clean.join(' ')
        end
      end
    end
  end
end

# ###############################################################################
# # Debug/Optimization/Warning build flags.
# ###############################################################################
# CCOPTFLAGS        ?=
# NVOPTFLAGS        ?=
# CCFLAGS           += -DDEBUG -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
# CCFLAGS_NC        += -DDEBUG -DHEMI_CUDA_DISABLE -Wall -Wextra -Wunknown-pragmas -Wno-system-headers -fdiagnostics-show-option
# # Always include DEBUG for nvcc so that errors caught by Hemi are always
# # reported.
# NVFLAGS           += -DDEBUG
# NVFLAGS_NC        += -x c++ -DDEBUG -DHEMI_CUDA_DISABLE

# # Need to suss out how to ignore system headers with llvm-g++, etc...
# # ifneq ($(DARWIN),)
# #   CCFLAGS         += -pedantic
# # endif
# ifeq ($(DEBUG),1)
#   CCFLAGS         += -g
#   CCFLAGS_NC      += -g
#   NVFLAGS         += -g -G
#   NVFLAGS_NC      += -g -G
# else
#   NVOPTFLAGS      += -O2
#   ifneq ($(DARWIN),)
#     CCOPTFLAGS    += -O4
#     CCOPTFLAGS_NC += -O4
#     NVOPTFLAGS_NC += -O4
#   else
#     CCOPTFLAGS    += -Os
#     CCOPTFLAGS_NC += -Os
#     NVOPTFLAGS_NC += -Os
#   endif
# endif


# ###############################################################################
# # Common includes and paths for CUDA, and CURAND.
# ###############################################################################
# CCINCLUDES    := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
# CCINCLUDES_NC := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
# NVINCLUDES    := -I $(CUDA_INC_PATH) -I $(SOURCE_PATH) -I $(SOURCE_PATH)/vendor
# NVINCLUDES_NC := -I$(CUDA_INC_PATH) -I$(SOURCE_PATH) -I$(SOURCE_PATH)/vendor
# LDFLAGS       += -lcudart -lcurand


# ###############################################################################
# # Allow Disabling CUDA.
# ###############################################################################
# ifneq ($(SKIP_CUDA),0)
#   NVCC       := $(GCC)
#   CCFLAGS    := $(CCFLAGS_NC)
#   NVFLAGS    := $(NVFLAGS_NC)
#   CCOPTFLAGS := $(CCOPTFLAGS_NC)
#   NVOPTFLAGS := $(NVOPTFLAGS_NC)
#   CCINCLUDES := $(CCINCLUDES_NC)
#   NVINCLUDES := $(NVINCLUDES_NC)
#   LDFLAGS    := $(LDFLAGS_NC)
# endif


# ###############################################################################
# # Apply optimization settings last...
# ###############################################################################
# ifneq ($(CCOPTFLAGS),)
#   CCFLAGS := $(CCFLAGS) $(CCOPTFLAGS)
# endif

# ifneq ($(NVOPTFLAGS),)
#   NVFLAGS := $(NVFLAGS) $(NVOPTFLAGS)
# endif


# ###############################################################################
# # Target rules.
# ###############################################################################

# ######################################
# # Test Cases
# ######################################
# $(INTERMEDIATE_PATH)/test/:
#   mkdir -p $(INTERMEDIATE_PATH)/test
#   mkdir -p $(BUILD_PATH)/test

# $(INTERMEDIATE_PATH)/test/hemi.o: $(SOURCE_PATH)/test/hemi.cu $(INTERMEDIATE_PATH)/test/
#   $(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

# $(BUILD_PATH)/test/hemi: $(INTERMEDIATE_PATH)/test/hemi.o $(INTERMEDIATE_PATH)/wie/device.o
#   $(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

# $(INTERMEDIATE_PATH)/test/randoms.o: $(SOURCE_PATH)/test/randoms.cpp $(INTERMEDIATE_PATH)/test/
#   $(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

# $(BUILD_PATH)/test/randoms: $(INTERMEDIATE_PATH)/test/randoms.o $(INTERMEDIATE_PATH)/wie/random.o $(INTERMEDIATE_PATH)/wie/device.o
#   $(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


# ######################################
# # Playground
# ######################################
# $(INTERMEDIATE_PATH)/etc/:
#   mkdir -p $(INTERMEDIATE_PATH)/etc
#   mkdir -p $(BUILD_PATH)/etc

# $(INTERMEDIATE_PATH)/etc/hello.o: $(SOURCE_PATH)/etc/hello.cu $(INTERMEDIATE_PATH)/etc/
#   $(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

# $(BUILD_PATH)/etc/hello: $(INTERMEDIATE_PATH)/etc/hello.o $(INTERMEDIATE_PATH)/wie/device.o $(INTERMEDIATE_PATH)/wie/stopwatch.o
#   $(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

# $(INTERMEDIATE_PATH)/etc/example.o: $(SOURCE_PATH)/etc/example.cu $(INTERMEDIATE_PATH)/etc/
#   $(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

# $(BUILD_PATH)/etc/example: $(INTERMEDIATE_PATH)/etc/example.o
#   $(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


# ######################################
# # Main Code
# ######################################
# $(INTERMEDIATE_PATH)/stars.o: $(SOURCE_PATH)/stars.cu
#   $(NVCC) $(NVFLAGS) $(NVINCLUDES) -o $@ -c $<

# $(BUILD_PATH)/stars: $(INTERMEDIATE_PATH)/stars.o
#   $(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


# ######################################
# # Support Code
# ######################################
# $(INTERMEDIATE_PATH)/wie/:
#   mkdir -p $(INTERMEDIATE_PATH)/wie

# $(INTERMEDIATE_PATH)/wie/device.o: $(SOURCE_PATH)/wie/device.cpp $(INTERMEDIATE_PATH)/wie/
#   $(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

# $(INTERMEDIATE_PATH)/wie/stopwatch.o: $(SOURCE_PATH)/wie/stopwatch.cpp $(INTERMEDIATE_PATH)/wie/
#   $(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<

# $(INTERMEDIATE_PATH)/wie/random.o: $(SOURCE_PATH)/wie/random.cpp $(INTERMEDIATE_PATH)/wie/
#   $(GCC) $(CCFLAGS) $(CCINCLUDES) -o $@ -c $<


# ######################################
# # Build Actions
# ######################################
# all: $(BUILD_PATH)/stars $(BUILD_PATH)/test/randoms $(BUILD_PATH)/test/hemi $(BUILD_PATH)/etc/hello $(BUILD_PATH)/etc/example

# clean:
#   rm -rf tmp/*

# clobber:
#   rm -rf bin/*

# # TODO: Better test harness, and unified control of flow via return code.
# test:
#   find $(TESTS_PATH) -type f -exec {} \;

# full_test:
#   make THOROUGH=1 test

# depend:
#   mkdep $(CCFLAGS) $(CCINCLUDES) $(shell find src -name '*.cpp')
#   mkdep -a $(NVFLAGS) $(NVINCLUDES) $(shell find src -name '*.cu')

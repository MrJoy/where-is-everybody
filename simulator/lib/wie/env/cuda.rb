module WIE
  module Env
    module CUDA
    private
      DEFAULT_PATHS={
        'Linux' => proc { '/usr/local/cuda' },
        'Darwin' => proc {
          [
            *FileList['/usr/local/cuda'],
            *(FileList['/Developer/NVIDIA/CUDA-*'].sort.reverse)
          ].extant.best
        }
      }

      LIB_SUFFIXES={
        'Linux' =>  { '_' => 'lib', 64 => 'lib64', },
        'Darwin' => { '_' => 'lib' },
      }

    public
      def self.auto_base_path
        @auto_base_path ||= DEFAULT_PATHS[WIE::Env::OS.kind].reify.simplify
      end

      def self.base_path
        @base_path ||= [
          ENV['CUDA_BASE_PATH'],
          self.auto_base_path
        ].extant.best.simplify
      end

      def self.lib_path
        @lib_path ||= begin
          suffix = LIB_SUFFIXES[WIE::Env::OS.kind][WIE::Env::OS.size]
          File.join(self.base_path, 'lib').simplify
        end
      end
    end
  end
end

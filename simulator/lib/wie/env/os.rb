module WIE
  module Env
    module OS
      def self.kind; @kind ||= (ENV['OS_KIND'] || `\\uname -s 2>/dev/null`).strip; end
      def self.arch; @arch ||= (ENV['OS_ARCH'] || `\\uname -m 2>/dev/null`.sub(/i386/, 'i686')).strip; end
      def self.size; @size ||= (ENV['OS_SIZE'] || self.arch.sub(/i.86/, '32').sub(/x86_64/, '64')).strip.to_i; end

      def self.linux?; @is_linux ||= (self.kind == 'Linux'); end
      def self.darwin?; @is_darwin ||= (self.kind == 'Darwin'); end
    end
  end
end

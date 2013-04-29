module WIE
  module Env
    module Mode
      def self.debug?; @debug ||= ([ENV['DEBUG'], 0].best.to_i != 0); end
    end
  end
end

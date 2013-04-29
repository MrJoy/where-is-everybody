module WIE
  module Debugging
    module Loader
      def self.init(ctx)
        puts "Initializing... #{ctx},  #{self}"
        @ctx_ref = ctx
        @ctx_ref.extend(WIE::Debugging::Loader)
        @ctx_ref.extend(WIE::Debugging::Reflection::ClassMethods)
      end

      protected

      class << self
      protected
        def context
          return @ctx_ref
        end
      end

      public

      def reload!
        # Hold on to our initial context, we'll need it later...
        ctx_ref_backup = WIE::Debugging::Loader.send(:context)

        # Disable all manner of warnings to avoid annoying cruft in the output.
        old_values = [$-v, $-w, $VERBOSE]
        $-v = $-w = $VERBOSE = nil

        # Unload rake tasks so changes in them and their dependencies are
        # properly reloaded.
        Rake::Task.clear

        # Unload all our code so old code doesn't stick around.
        Object.class_eval do
          remove_const :WIE rescue nil
        end

        # Tell Ruby to go ahead and re-require all our code.  The File.join
        # ensures we have a trailing slash, just for the sake of being super
        # paranoid.
        base_dir = File.join(BASE_DIR, '')
        $LOADED_FEATURES.reject! do |feature|
          feature.start_with?(base_dir)
        end

        # Reload our code.
        load 'Rakefile'

        # Re-load this code, and re-run extend so that the new version of this
        # method shows up.
        require 'wie/debugging'
        WIE::Debugging::Loader.init(ctx_ref_backup)

        # Finally, re-enable warnings.
        $-v, $-w, $VERBOSE = old_values

        return nil
      end
    end
  end
end

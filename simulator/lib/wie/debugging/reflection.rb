module WIE
  module Debugging
    module Reflection
      def self.included(klass)
        klass.extend ClassMethods
      end

      module ClassMethods
        # Helper to reflect on a module.
        #
        # 1. Find public, static methods on the given module that were NOT
        #    inherited.
        # 2. Sort them such that predicate methods (`foo?`) come last, and both
        #    groups of methods (predicates and non-predicates) are each sorted by
        #    name.
        # 3. Call the method, and get an array of [name, value] pairs.
        # 4. Find sub-modules.
        # 5. Recurse on sub-modules.
        # 6. Return a structure containing all of this data.
        def reflect(mref, dive_into=false, dive_path='')
          full_name = mref.to_s
          module_name = full_name.split(/::/).last

          if(dive_into && !full_name.start_with?(dive_path))
            return nil if(!dive_path.start_with?(full_name))
            return [
              module_name,
              {
                methods: [],
                predicates: [],
                modules: get_modules(mref, dive_into, dive_path),
              }
            ]
          end

          method_info = (Set.new(mref.singleton_methods) & Set.new(mref.public_methods)).
            to_a.
            select { |m| mref.method(m).arity == 0 }.
            map(&:to_s).
            map { |m| [m.sub(/[^\?]+/, ''), m] }.
            sort. # Put predicate methods ('foo?' last...) first, then sort by name.
            map { |m| m.last }.
            map(&:to_sym).
            map { |m| [m, mref.send(m)] }

          normal_methods = method_info.select { |m| m[0] !~ /\?$/ }
          predicates = method_info.select { |m| m[0] =~ /\?$/ }
          modules = get_modules(mref, dive_into, dive_path)

          return [
            module_name,
            {
              methods: normal_methods,
              predicates: predicates,
              modules: modules,
            },
          ]
        end

        def show(mref, indentation = 0)
          if(mref.is_a?(Module))
            name_pieces = mref.to_s.split(/::/)
            if(name_pieces.length > 1)
              mref = Module.const_get(name_pieces.first)
              (name, data) = *WIE::Debugging::Reflection.reflect(mref, true, name_pieces.join('::'))
            else
              (name, data) = *WIE::Debugging::Reflection.reflect(mref)
            end
            # name = mref.to_s.split(/::/)
            # indentation = name.length - 1
            # name = name.join("::")
          else
            (name, data) = *mref
          end

          name_prefix = "#" * (indentation+1)
          item_prefix = "* "

          (num_m, num_p, num_s) = *[data[:methods].length, data[:predicates].length, data[:modules].length]
          if(num_m+num_p+num_s > 0)
            puts "#{name_prefix} #{name}"

            list(data, :methods, item_prefix)
            list(data, :predicates, item_prefix)

            if(num_s > 0)
              data[:modules].each_with_index do |submodule_data, idx|
                puts if(idx <= num_s-1)
                show(submodule_data, indentation+1)
              end
            end
          end
          puts if(indentation == 0)

          return nil
        end

      protected

        def list(items, subset, prefix)
          if(items[subset].length > 0)
            puts
            window = 3 + [*items[:methods],*items[:predicates]].
              map(&:first).
              map(&:length).
              max
            items[subset].each do |(name,val)|
              puts ("%s%-#{window}s%s" % [prefix, "#{name}:", val])
            end
          end
        end

        def get_modules(mref, dive_into, dive_path)
          return mref.
            constants.
            sort.
            map { |name| mref.const_get(name) }.
            select { |const| const.is_a?(Module) }.
            map { |mod| reflect(mod, dive_into, dive_path) }.
            reject { |data| data.nil? }.
            reject do |data|
              data[1][:methods].length == 0 &&
              data[1][:predicates].length == 0 &&
              data[1][:modules].length == 0
            end
        end
      end
      extend ClassMethods
    end
  end
end

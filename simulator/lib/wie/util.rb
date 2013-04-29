class Array
  def clean
    return self.reject { |val| val.blank? }.compact
  end

  def extant
    return self.clean.select { |path| File.exist?(path) }
  end

  def best
    return self.clean.map(&:reify).first
  end

  def blank?; return self.count == 0; end
end

class Hash
  def best(key)
    puts "<<<<"
    puts self.inspect
    puts key
    puts ">>>>"
    return [
      self[key],
      self['_']
    ].best
  end

  def blank?; return self.keys.count == 0; end
end

class String
  def which
    tmp = `\\which #{self}`.strip
    tmp = nil if(tmp.empty?)
    return tmp
  end

  def blank?; return self == ''; end

  def simplify; return self.sub(/\/$/, ''); end
end

class NilClass
  def which; return self; end

  def blank?; return true; end

  def reify; return self; end

  def simplify; return self; end
end

class Object
  def blank?; return false; end

  def reify; return self; end
end

class Proc
  def reify; return self.call; end
end

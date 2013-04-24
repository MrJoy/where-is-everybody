#ifndef WIE_FILE_H
#define WIE_FILE_H

#pragma once


///////////////////////////////////////////////////////////////////////////////
// System includes.
///////////////////////////////////////////////////////////////////////////////
#include <cassert>
#include <string>
// #include <sstream>
// #include <stdexcept>


///////////////////////////////////////////////////////////////////////////////
// Class definition.
///////////////////////////////////////////////////////////////////////////////
namespace WIE {
  template< typename T >
  class File
  {
  public:
    File(const std::string& pFilename);
    ~File();

    void save(T* buffer, size_t records);
    void load(T* buffer, size_t records);


  private:
    const std::string&  filename;

    // NOT IMPLEMENTED!
    File(File& src);
    File(const File& src);
    File(volatile File& src);
    File(const volatile File& src);
  };
}

#endif

#include "wie/file.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction.
///////////////////////////////////////////////////////////////////////////////
WIE::File::File(const std::string& pFilename)
  : filename(pFilename)
{
}

WIE::File::~File()
{
}


///////////////////////////////////////////////////////////////////////////////
// Main class logic.
///////////////////////////////////////////////////////////////////////////////
void WIE::File::save(T* buffer, size_t records)
{
  int filesize = records * sizeof(T);
  int result;

  // TODO: Obey umask?
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0644);
  assert(fd != -1);

  // Stretch the file size to the size of the (mmapped) array of ints.
  result = lseek(fd, filesize-1, SEEK_SET);
  assert(result != -1);

  // Something needs to be written at the end of the file to have the file
  // actually have the new size.  Just writing an empty string at the current
  // file position will do.
  //
  // Note:
  //  - The current position in the file is at the end of the stretched
  //    file due to the call to lseek().
  //  - An empty string is actually a single '\0' character, so a zero-byte
  //    will be written at the last byte of the file.
  result = write(fd, "", 1);
  assert(result != -1);

  // Now the file is ready to be mmapped.
  T* buffer = mmap(buffer, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(map != MAP_FAILED);

  // Don't forget to free the mmapped memory.
  reult = munmap(buffer, filesize);
  assert(munmap(buffer, filesize) != -1);

  // Un-mmaping doesn't close the file, so we still need to do that.
  close(fd);
}

void WIE::File::load(T* buffer, size_t records)
{
/*
  int i;
  int fd;
  int *map;  // mmapped array of int's

  fd = open(FILEPATH, O_RDONLY);
  if (fd == -1) {
    perror("Error opening file for reading");
    exit(EXIT_FAILURE);
  }

  map = mmap(0, FILESIZE, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    perror("Error mmapping the file");
    exit(EXIT_FAILURE);
  }

  // Read the file int-by-int from the mmap
  for (i = 1; i <=NUMINTS; ++i) {
    printf("%d: %d\n", i, map[i]);
  }

  if (munmap(map, FILESIZE) == -1) {
    perror("Error un-mmapping the file");
  }
  close(fd);
*/
}

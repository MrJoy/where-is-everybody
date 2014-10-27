#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

unsigned int
generate_seed()
{
  FILE* randomSource = fopen("/dev/urandom", "rb");
  unsigned int seed;
  int bytes_to_read = static_cast<signed int>(sizeof(unsigned int));
  int items_read = fread( &seed, bytes_to_read, 1, randomSource );
  fclose( randomSource );
  if(items_read != 1){
    fprintf(stderr, "did not read a full unsigned int.");
    exit(1);
  }
  return seed;
}

typedef uint8_t output_t ;
#define STAR_STATE_CHAR(x) (int)((output_t)('0') + x)

void
inspect( const char prefix, const output_t *array, const int line, const int n )
{
  for( int i=0; i<n; ++i ) {
    if( i % line == 0 ){
      putchar(prefix);
    }
    putchar(STAR_STATE_CHAR(array[i]));
    if( (i + 1) % line == 0 ){
      putchar('\n');
    }
  }
}




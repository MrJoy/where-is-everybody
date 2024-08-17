#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "stars_host_device_shared.h"

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

void
show_counters(int copy_ret, int iteration, const counter_t *counters)
{
  printf("%i: %u %u %u %u %u %u %u %u, %i\n", iteration,
      counters[0], counters[1], counters[2], counters[3],
      counters[4], counters[5], counters[6], counters[7], copy_ret);
}

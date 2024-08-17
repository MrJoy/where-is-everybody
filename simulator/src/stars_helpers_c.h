#ifndef STARS_HELPERS_C
#define STARS_HELPERS_C 1
#include "stars_config.h"
extern unsigned int generate_seed();
extern void inspect( const char prefix, const output_t *array, const int line, const int n );
extern void show_counters(int copy_ret, int iteration, const counter_t *counters);
#endif

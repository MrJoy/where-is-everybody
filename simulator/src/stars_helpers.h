#ifndef _STAR_HELPERS_
#define _STAR_HELPERS_ 1
//required for output_t and counter_t typedefs, uint8_t and uint32_t
#include "stars_host_device_shared.h"

__global__ void init_rands(unsigned int seed, curandStateXORWOW_t *rgens );

__global__ void init_buf(output_t *outs);
__global__ void init_counters(counter_t *buf);


__global__ void iterate_states(
    curandStateXORWOW_t *rgens,
    output_t *buf_in,
    output_t *buf_out);

__global__ void count_states(output_t *buf, counter_t *counters);
__global__ void sum_states(counter_t *counters, counter_t *sums);

#endif /* _STAR_HELPERS_ 1 */

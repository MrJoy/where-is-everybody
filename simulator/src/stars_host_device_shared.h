#ifndef _STARS_HOST_DEVICE_SHARED_H_
#define _STARS_HOST_DEVICE_SHARED_H_ 1

#include <stdint.h>
typedef uint8_t output_t ;
typedef uint32_t counter_t ;

// 2**10
#define THREADS_PER_BLOCK 1024
// 2**7
#define BLOCKS 128
// 2**17
#define THREADS_EVER (THREADS_PER_BLOCK * BLOCKS)
// 2**30
#define STARS (1024 * 1024 * 1024)
// 2**30 / 2**17 = 2**13
#define STARS_PER_THREAD (STARS / THREADS_EVER)

#define NUM_STATES 8

#endif /* _STARS_HOST_DEVICE_SHARED_H_ */

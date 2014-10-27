//TODO: move me.
typedef uint8_t output_t ;

#ifndef _STAR_HELPERS_
#define _STAR_HELPERS_ 1

__global__ void init_rands(unsigned int seed, curandStateXORWOW_t *rgens );

__global__ void init_buf( output_t *outs, output_t value, int n);

__global__ void iterate_states(
    curandStateXORWOW_t *rgens,
    output_t *buf_in,
    output_t *buf_out,
    int neighborhood_stars,
    output_t *state_matrix,
    float *pchange
    );

#endif /* _STAR_HELPERS_ 1 */

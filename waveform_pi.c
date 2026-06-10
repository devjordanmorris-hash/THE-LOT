// waveform_pi.c — Integer CORDIC-like sine generator for Pi3B (CSV output)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// Fixed-point Q31 angle/input, Q15 output amplitude for easy CSV
static inline int32_t isine_q31_q15(uint32_t theta_q31){
    // Simple 3rd-order Bhaskara-style polynomial approximation, integerized
    // x in [-pi, pi] mapped from theta_q31
    // This is for demo; for production use a proper CORDIC or table+interp.
    const double pi = 3.14159265358979323846;
    double x = ((int64_t)theta_q31 - 0x80000000LL) * (pi / 2147483648.0); // [-pi, pi)
    double y = x * (1.2732395447351628 - 0.4052847345693511 * (x<0?-x:x));
    int32_t q15 = (int32_t)lrint( y * 32767.0 );
    if (q15>32767) q15=32767; if (q15<-32768) q15=-32768;
    return q15;
}

int main(int argc, char** argv){
    size_t N = 65536;
    if (argc>1){ size_t v=strtoull(argv[1],NULL,10); if (v>=16 && v<=10000000) N=v; }
    FILE* f = fopen("waveform.csv","w");
    if (!f){ perror("waveform.csv"); return 1; }
    fprintf(f,"n,theta_q31,sin_q15\n");
    uint32_t step = (uint32_t)(0xFFFFFFFFu / N);
    uint32_t t=0;
    for (size_t n=0;n<N;++n){
        int32_t s = isine_q31_q15(t);
        fprintf(f,"%zu,%u,%d\n", n, t, s);
        t += step;
    }
    fclose(f);
    printf("Wrote waveform.csv with %zu samples.\n", N);
    return 0;
}

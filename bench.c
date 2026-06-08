#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

static inline float fast_wrap_pi(float x){
    const float inv2pi = 0.15915494309189535f;
    float k = floorf(x * inv2pi + 0.5f);
    return x - k * 2.0f * (float)M_PI;
}

static inline float my_sinf(float x){
    x = fast_wrap_pi(x);
    const float c2 = -0.16666657f;
    const float c4 =  0.00830636f;
    float x2 = x*x;
    return x * (1.0f + c2*x2 + c4*x2*x2);
}

static void matmul4_baseline(const float *A, const float *B, float *C){
    for(int i=0;i<4;i++)
      for(int j=0;j<4;j++){
        float s=0.f;
        for(int k=0;k<4;k++) s += A[i*4+k]*B[k*4+j];
        C[i*4+j]=s;
      }
}

#if defined(__aarch64__)
#include <arm_neon.h>
static void matmul4_neon(const float *A, const float *B, float *C){
    float32x4_t b0 = vld1q_f32(B+0);
    float32x4_t b1 = vld1q_f32(B+4);
    float32x4_t b2 = vld1q_f32(B+8);
    float32x4_t b3 = vld1q_f32(B+12);
    for(int i=0;i<4;i++){
        float32x4_t a = vdupq_n_f32(A[i*4+0]);
        float32x4_t r = vmulq_f32(a, b0);
        a = vdupq_n_f32(A[i*4+1]); r = vmlaq_f32(r, a, b1);
        a = vdupq_n_f32(A[i*4+2]); r = vmlaq_f32(r, a, b2);
        a = vdupq_n_f32(A[i*4+3]); r = vmlaq_f32(r, a, b3);
        vst1q_f32(C+i*4, r);
    }
}
#else
static void matmul4_neon(const float *A, const float *B, float *C){
    matmul4_baseline(A,B,C);
}
#endif

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int main(int argc, char**argv){
    size_t N = (argc>1)? strtoull(argv[1],0,10) : 50000000ULL;
    volatile float sink = 0.f;
    double t0 = now_s();
    for(size_t i=0;i<N;i++){
        float x = (float)i * 0.0001f;
        sink += sinf(x);
    }
    double t1 = now_s();
    for(size_t i=0;i<N;i++){
        float x = (float)i * 0.0001f;
        sink += my_sinf(x);
    }
    double t2 = now_s();

    float A[16], B[16], C[16];
    for(int i=0;i<16;i++){ A[i]=(i%5)+0.1f; B[i]=(i%7)+0.2f; }
    int repeats = 5000000;
    double t3 = now_s();
    for(int r=0;r<repeats;r++) matmul4_baseline(A,B,C);
    double t4 = now_s();
    for(int r=0;r<repeats;r++) matmul4_neon(A,B,C);
    double t5 = now_s();

    printf("results(sink)=%f\n", sink);
    printf("sinf      : %.3f s  (%.2f M eval/s)\n", t1-t0, N/1e6/(t1-t0));
    printf("my_sinf   : %.3f s  (%.2f M eval/s)\n", t2-t1, N/1e6/(t2-t1));
    printf("4x4 base  : %.3f s  (%.2f M mul/s)\n", t4-t3, repeats/1e6/(t4-t3));
    printf("4x4 NEON  : %.3f s  (%.2f M mul/s)\n", t5-t4, repeats/1e6/(t5-t4));
    return 0;
}

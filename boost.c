// boost — Tier0 CPU lab (safe, local)
// Build: clang -O3 boost -o boost -lpthread
// Run:   ./boost --n=64 --threads=0   (0 = auto threads)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

// ---------------- timing ----------------
static double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void print_time(const char* tag, double t0, double t1){
    double s = t1 - t0; printf("%s,sec=%.9f,ms=%.3f\n", tag, s, s*1e3);
}

// ---------------- args ----------------
static int MATN = 64; // matrix size
static int THREADS = 0; // 0 = auto
static void parse_args(int argc, char** argv){
    for(int i=1;i<argc;i++){
        if(strncmp(argv[i],"--n=",4)==0){ MATN = atoi(argv[i]+4); if(MATN<8) MATN=8; }
        else if(strncmp(argv[i],"--threads=",10)==0){ THREADS = atoi(argv[i]+10); }
    }
}
static int cpu_threads(void){ long n = sysconf(_SC_NPROCESSORS_ONLN); if(n<1) n=1; return (int)n; }

// ---------------- RNG ----------------
static uint64_t splitmix64(uint64_t *x){
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ---------------- bitwise add/mul ----------------
static uint64_t add_bitwise(uint64_t a, uint64_t b){
    while(b){ uint64_t c=a&b; a^=b; b=c<<1; } return a;
}
static uint64_t mul_bitwise(uint64_t a, uint64_t b){
    uint64_t r=0; while(b){ if(b&1ull) r=add_bitwise(r,a); a<<=1; b>>=1; } return r;
}

// ---------------- 128-bit helpers and Karatsuba ----------------
typedef struct { uint64_t lo, hi; } u128;
static u128 add128(u128 x, u128 y){ u128 z; uint64_t lo=x.lo+y.lo; z.lo=lo; z.hi=x.hi+y.hi+(lo<x.lo); return z; }
static u128 shl128(u128 x, unsigned s){ s&=127u; if(!s) return x; if(s>=64){ u128 z={0, x.lo<<(s-64)}; return z; } u128 z; z.hi=(x.hi<<s)|(x.lo>>(64-s)); z.lo=x.lo<<s; return z; }
static u128 mul64_128(uint64_t a, uint64_t b){ __uint128_t p=(__uint128_t)a*(__uint128_t)b; u128 z={(uint64_t)p,(uint64_t)(p>>64)}; return z; }
static u128 karatsuba128(uint64_t a1, uint64_t a0, uint64_t b1, uint64_t b0){
    u128 z0=mul64_128(a0,b0); u128 z2=mul64_128(a1,b1);
    __uint128_t t=(__uint128_t)(a1+a0)*(__uint128_t)(b1+b0);
    __uint128_t Z0=((__uint128_t)z0.hi<<64)|z0.lo, Z2=((__uint128_t)z2.hi<<64)|z2.lo;
    __uint128_t Z1=t-Z0-Z2; u128 m={ (uint64_t)Z1,(uint64_t)(Z1>>64) };
    return add128(shl128(z2,128), add128(shl128(m,64), z0));
}

// ---------------- 8x8 product LUT and LUT-based 64-bit mul ----------------
static uint16_t MUL8[256][256]; static int MUL8_INIT=0;
static void init_mul8(void){ if(MUL8_INIT) return; for(int a=0;a<256;a++) for(int b=0;b<256;b++) MUL8[a][b]=(uint16_t)((uint16_t)a*(uint16_t)b); MUL8_INIT=1; }
static u128 add128_bitwise(u128 x, u128 y){
    // low 64 via bitwise adder
    uint64_t a=x.lo, b=y.lo; while(b){ uint64_t c=a&b; a^=b; b=c<<1; }
    uint64_t lo=a; uint64_t carry=(lo<x.lo);
    // high add with carry using bitwise again
    uint64_t ah=x.hi+carry, bh=y.hi; while(bh){ uint64_t c=ah&bh; ah^=bh; bh=c<<1; }
    u128 z={lo,ah}; return z;
}
static u128 acc_shift_u16(u128 acc, uint16_t val, int byte_shift){
    int s=byte_shift*8; u128 part={0,0};
    if(s<64){ part.lo=((uint64_t)val)<<s; part.hi=((uint64_t)val)>>(64-s); }
    else    { part.hi=((uint64_t)val)<<(s-64); }
    return add128_bitwise(acc, part);
}
static uint64_t mul_lut8_64(uint64_t A, uint64_t B){
    init_mul8(); u128 acc={0,0}; uint8_t a[8], b[8];
    for(int i=0;i<8;i++){ a[i]=(uint8_t)(A>>(8*i)); b[i]=(uint8_t)(B>>(8*i)); }
    for(int i=0;i<8;i++) for(int j=0;j<8;j++) acc=acc_shift_u16(acc, MUL8[a[i]][b[j]], i+j);
    return acc.lo; // low 64 result
}

// ---------------- scalar benches ----------------
static void bench_scalar64(size_t iters){
    uint64_t seed=7, sink=0; double t0=now_sec();
    for(size_t i=0;i<iters;i++){ uint64_t a=splitmix64(&seed), b=splitmix64(&seed); sink^=a*b; }
    double t1=now_sec();
    for(size_t i=0;i<iters;i++){ uint64_t a=splitmix64(&seed), b=splitmix64(&seed); sink^=mul_bitwise(a,b); }
    double t2=now_sec();
    for(size_t i=0;i<iters;i++){ uint64_t a=splitmix64(&seed), b=splitmix64(&seed); sink^=mul_lut8_64(a,b); }
    double t3=now_sec();
    printf("bits,method,avg_us\n");
    printf("64,builtin_mul,%.3f\n", (t1-t0)*1e6/iters);
    printf("64,bitwise_shift_add,%.3f\n", (t2-t1)*1e6/iters);
    printf("64,lut8_schoolbook,%.3f\n", (t3-t2)*1e6/iters);
    (void)sink;
}
static void bench_karatsuba128(size_t iters){
    uint64_t seed=13, sink=0; double t0=now_sec();
    for(size_t i=0;i<iters;i++){
        uint64_t a1=splitmix64(&seed), a0=splitmix64(&seed);
        uint64_t b1=splitmix64(&seed), b0=splitmix64(&seed);
        u128 r=karatsuba128(a1,a0,b1,b0); sink^=r.lo^r.hi;
    }
    double t1=now_sec();
    printf("128,karatsuba_halves,%.3f us\n", (t1-t0)*1e6/iters);
}

// ---------------- matrix helpers ----------------
static void mat_fill(uint64_t *M, uint64_t *seed){ for(int i=0;i<MATN*MATN;i++) M[i]=splitmix64(seed)&0xFFFFull; }
static void mat_zero(uint64_t *M){ memset(M,0,sizeof(uint64_t)*MATN*MATN); }
static uint64_t mat_checksum(const uint64_t *C){ uint64_t x=0; for(int i=0;i<MATN*MATN;i++){ x ^= C[i] + 0x9e3779b97f4a7c15ull*(i+1); } return x; }

static void matmul_builtin(const uint64_t *A,const uint64_t *B,uint64_t *C){
    for(int i=0;i<MATN;i++){
        for(int j=0;j<MATN;j++){
            uint64_t s=0; for(int k=0;k<MATN;k++) s += A[i*MATN+k]*B[k*MATN+j]; C[i*MATN+j]=s;
        }
    }
}
static void matmul_bitwise(const uint64_t *A,const uint64_t *B,uint64_t *C){
    for(int i=0;i<MATN;i++){
        for(int j=0;j<MATN;j++){
            uint64_t s=0; for(int k=0;k<MATN;k++){ uint64_t p=mul_bitwise(A[i*MATN+k],B[k*MATN+j]); s=add_bitwise(s,p);} C[i*MATN+j]=s;
        }
    }
}
static void matmul_lut8(const uint64_t *A,const uint64_t *B,uint64_t *C){
    init_mul8();
    for(int i=0;i<MATN;i++){
        for(int j=0;j<MATN;j++){
            uint64_t s=0; for(int k=0;k<MATN;k++){ uint64_t p=mul_lut8_64(A[i*MATN+k],B[k*MATN+j]); s += p; } C[i*MATN+j]=s;
        }
    }
}

// ---------------- multithreaded matmul ----------------
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

typedef struct { const uint64_t *A,*B; uint64_t *C; int r0,r1; int variant; } mm_task_t; // 0=builtin,1=bitwise,2=lut8

static void *mm_worker(void *arg){
    mm_task_t *t=(mm_task_t*)arg; int r0=t->r0, r1=t->r1;
    if(t->variant==0){
        for(int i=r0;i<r1;i++) for(int j=0;j<MATN;j++){ uint64_t s=0; for(int k=0;k<MATN;k++) s+=t->A[i*MATN+k]*t->B[k*MATN+j]; t->C[i*MATN+j]=s; }
    } else if(t->variant==1){
        for(int i=r0;i<r1;i++) for(int j=0;j<MATN;j++){ uint64_t s=0; for(int k=0;k<MATN;k++){ uint64_t p=mul_bitwise(t->A[i*MATN+k],t->B[k*MATN+j]); s=add_bitwise(s,p);} t->C[i*MATN+j]=s; }
    } else {
        for(int i=r0;i<r1;i++) for(int j=0;j<MATN;j++){ uint64_t s=0; for(int k=0;k<MATN;k++){ uint64_t p=mul_lut8_64(t->A[i*MATN+k],t->B[k*MATN+j]); s+=p;} t->C[i*MATN+j]=s; }
    }
    return NULL;
}

static void bench_matmul_mt(int variant, int threads){
    if(threads<=0) threads=cpu_threads(); if(threads>MATN) threads=MATN; if(threads>MAX_THREADS) threads=MAX_THREADS;
    uint64_t *A=malloc(sizeof(uint64_t)*MATN*MATN), *B=malloc(sizeof(uint64_t)*MATN*MATN), *C=malloc(sizeof(uint64_t)*MATN*MATN);
    uint64_t seed=123; mat_fill(A,&seed); mat_fill(B,&seed); mat_zero(C);
    pthread_t th[MAX_THREADS]; mm_task_t tasks[MAX_THREADS];
    int rows_per=MATN/threads, rem=MATN%threads, start=0;
    double t0=now_sec();
    for(int ti=0; ti<threads; ++ti){ int extra=(ti<rem)?1:0; int end=start+rows_per+extra; tasks[ti]=(mm_task_t){A,B,C,start,end,variant}; pthread_create(&th[ti],NULL,mm_worker,&tasks[ti]); start=end; }
    for(int ti=0; ti<threads; ++ti) pthread_join(th[ti], NULL);
    double t1=now_sec();
    const char* name = (variant==0?"builtin_matmul": (variant==1?"bitwise_matmul":"lut8_matmul"));
    printf("mat%d_threads%d,%s,%.6f,checksum=0x%016llx\n", MATN, threads, name, (t1-t0), (unsigned long long)mat_checksum(C));
    free(A); free(B); free(C);
}

// ---------------- main ----------------
int main(int argc, char** argv){
    parse_args(argc, argv);
    printf("MATN=%d, threads=%d (0=auto)\n", MATN, THREADS);

    // scalar
    bench_scalar64(200000);
    bench_karatsuba128(200000);

    // single-thread matrices
    uint64_t *A=malloc(sizeof(uint64_t)*MATN*MATN), *B=malloc(sizeof(uint64_t)*MATN*MATN), *C=malloc(sizeof(uint64_t)*MATN*MATN);
    uint64_t seed=321; mat_fill(A,&seed); mat_fill(B,&seed); mat_zero(C);

    double t0=now_sec(); matmul_builtin(A,B,C); double t1=now_sec();
    printf("mat%d,builtin_matmul,%.6f,checksum=0x%016llx\n", MATN, (t1-t0), (unsigned long long)mat_checksum(C)); mat_zero(C);

    double t2=now_sec(); matmul_bitwise(A,B,C); double t3=now_sec();
    printf("mat%d,bitwise_matmul,%.6f,checksum=0x%016llx\n", MATN, (t3-t2), (unsigned long long)mat_checksum(C)); mat_zero(C);

    double t4=now_sec(); matmul_lut8(A,B,C); double t5=now_sec();
    printf("mat%d,lut8_matmul,%.6f,checksum=0x%016llx\n", MATN, (t5-t4), (unsigned long long)mat_checksum(C));

    free(A); free(B); free(C);

    // multithreaded (auto threads unless overridden)
    int t = (THREADS>0? THREADS : cpu_threads());
    bench_matmul_mt(0, t);
    bench_matmul_mt(1, t);
    bench_matmul_mt(2, t);

    return 0;
}

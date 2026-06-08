// jlo3_portable.c — portable J-LO triangles + zlib (patch: dq clamp + idx guard)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#define SCALE (1 << 15)
#define N 200000

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (double)ts.tv_sec*1000.0 + (double)ts.tv_nsec/1.0e6; }

static void make_signal(float *x){
  const double f1=13.0,f2=4.81;
  for(int i=0;i<N;i++){ double t=(double)i/(double)N;
    x[i]=(float)(sin(2.0*M_PI*f1*t)*0.7 + 0.25*sin(2.0*M_PI*f2*t+0.6)); }
}

static void float_to_fix(const float *src,int32_t *dst){
  for(int i=0;i<N;i++) dst[i]=(int32_t)llround((double)src[i]*SCALE);
}
static void fix_to_float(const int32_t *src,float *dst){
  for(int i=0;i<N;i++) dst[i]=(float)src[i]/(float)SCALE;
}
static void rms_and_max(const float *a,const float *b,double *rms,double *maxabs){
  double s=0,m=0; for(int i=0;i<N;i++){ double e=(double)a[i]-b[i]; s+=e*e; if(fabs(e)>m)m=fabs(e); }
  *rms=sqrt(s/(double)N); *maxabs=m;
}

typedef struct{ int i0; int32_t v0; int i1; int32_t v1; } Segment;
static inline int64_t iabs64(int64_t x){ return x>=0?x:-x; }

/* ---- PATCH 1: robust maxDev with step + strict interior idx ---- */
static void maxDev_step(const int32_t *a,int i0,int32_t v0,int i1,int32_t v1,
                        int step,int *bestIdx,int64_t *bestErr){
  int n=i1-i0; *bestIdx=i0; *bestErr=0; if(n<=1) return;
  int64_t dv=(int64_t)v1-(int64_t)v0;
  for(int i=i0+1;i<i1;i+=step){
    int64_t num=(int64_t)v0*n + dv*(i-i0);
    int64_t err=iabs64((int64_t)a[i]*n - num);
    if(err>*bestErr){ *bestErr=err; *bestIdx=i; }
  }
  if(*bestIdx<=i0 || *bestIdx>=i1){             // <-- ensure split point
    *bestIdx = i0 + ((i1 - i0) >> 1);
    if(*bestIdx<=i0) *bestIdx=i0+1;
    if(*bestIdx>=i1) *bestIdx=i1-1;
  }
}

static void compJLO_fast(const int32_t *a,int32_t eps,int step,
                         Segment **outSegs,int *outCount){
  Segment *stack=(Segment*)malloc(sizeof(Segment)*N);
  Segment *segs =(Segment*)malloc(sizeof(Segment)*N);
  int sp=0, sc=0; stack[sp++]=(Segment){0,a[0],N-1,a[N-1]};
  while(sp>0){
    Segment s=stack[--sp]; int i0=s.i0, i1=s.i1, n=i1-i0; int32_t v0=s.v0, v1=s.v1;
    if(n<=1){ segs[sc++]=s; continue; }
    int idx; int64_t err; maxDev_step(a,i0,v0,i1,v1,step,&idx,&err);
    if(err <= (int64_t)eps * n){ segs[sc++]=s; }
    else { int32_t vm=a[idx]; stack[sp++]=(Segment){idx,vm,i1,v1}; stack[sp++]=(Segment){i0,v0,idx,vm}; }
  }
  *outSegs=segs; *outCount=sc; free(stack);
}

static void reconJLO(const Segment *segs,int count,int32_t *y){
  for(int s=0;s<count;s++){
    int i0=segs[s].i0,i1=segs[s].i1,n=i1-i0; int32_t v0=segs[s].v0,v1=segs[s].v1;
    if(n<=0){ if(i0>=0 && i0<N) y[i0]=v0; continue; }
    int64_t dv=(int64_t)v1-(int64_t)v0;
    for(int i=i0;i<=i1;i++){ int64_t t=i-i0; y[i]=(int32_t)((int64_t)v0 + (dv*t)/n); }
  }
}

static void quantRes(const int32_t *o,const int32_t *r,int32_t dq,int16_t *out){
  for(int i=0;i<N;i++){
    int v=(o[i]-r[i])/dq;
    if(v<-32768)v=-32768; else if(v>32767)v=32767;
    out[i]=(int16_t)v;
  }
}

/* zlib baseline */
static void bench_zlib(const uint8_t *input,size_t len){
  uLongf bound=compressBound((uLong)len);
  uint8_t *dst=(uint8_t*)malloc(bound);
  double t0=now_ms(); uLongf outSize=bound;
  int rc=compress2(dst,&outSize,input,(uLong)len,Z_BEST_SPEED);
  double enc=now_ms()-t0;
  uint8_t *dec=(uint8_t*)malloc(len);
  t0=now_ms(); uLongf decLen=(uLong)len; int rc2=uncompress(dec,&decLen,dst,outSize);
  double decT=now_ms()-t0;
  if(rc!=Z_OK||rc2!=Z_OK||decLen!=(uLong)len) fprintf(stderr,"zlib error\n");
  printf("zlib  ratio %.3f  enc %.2f ms  dec %.2f ms\n",(double)outSize/(double)len,enc,decT);
  free(dst); free(dec);
}

int main(void){
  float *x=(float*)malloc(sizeof(float)*N);
  make_signal(x);
  size_t raw=N*sizeof(float);
  printf("\n=== Standard Compressor (zlib-only, portable) ===\n");
  bench_zlib((const uint8_t*)x, raw);

  int32_t *xFix=(int32_t*)malloc(sizeof(int32_t)*N);
  float_to_fix(x,xFix);

  /* ---- PATCH 2: clamp dq >= 1 to avoid divide-by-zero ---- */
  int32_t eps=(int32_t)llround(1e-4 * SCALE);
  int32_t dq =(int32_t)llround(1e-5 * SCALE);
  if(dq < 1) dq = 1;

  int step=4; /* try 1 (precise), 2, 4, 8 (faster) */

  double t0=now_ms();
  Segment *segs; int segCount=0;
  compJLO_fast(xFix, eps, step, &segs, &segCount);
  double t1=now_ms();

  int32_t *base=(int32_t*)calloc(N,sizeof(int32_t));
  reconJLO(segs, segCount, base);
  double t2=now_ms();

  int16_t *q=(int16_t*)malloc(sizeof(int16_t)*N);
  quantRes(xFix, base, dq, q);
  double t3=now_ms();

  float *y=(float*)malloc(sizeof(float)*N);
  fix_to_float(base,y);
  double rms,maxA; rms_and_max(x,y,&rms,&maxA);

  double encMs=t3-t0;
  double ratio=(16.0 + segCount*16.0 + N*2.0)/ (double)raw;

  printf("\nseg %.2f ms  recon %.2f ms  quant %.2f ms\n", t1-t0, t2-t1, t3-t2);
  printf("\n=== J-LO Triangle Engine (portable) ===\n");
  printf("ratio %.3f  enc %.2f ms  RMS %.2e  Max %.2e  segments %d (avg span ~%.1f)\n",
         ratio, encMs, rms, maxA, segCount, (double)N/(double)segCount);

  free(segs); free(base); free(q); free(y); free(xFix); free(x);
  return 0;
}

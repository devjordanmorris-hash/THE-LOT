// jlo_demo.c — Swift J-L0 triangle compressor, rewritten in portable C (zlib-only)
//
// Build:  cc -O3 jlo_demo.c -o jlo_demo -lz
// Runs on macOS/Linux. Produces JLO/CSV/RAW files, zlib baselines, ε-sweep CSV + HTML.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <zlib.h>   // link with -lz

#define JLO2_SCALE 32767.0f  // int16 quant for v0/v1 (assumes |x|<=~1)

static inline uint32_t zigzag32(int32_t v){ return (v<<1) ^ (v>>31); }

static uint8_t* write_uvar32(uint8_t *p, uint32_t v){
    while(v >= 0x80){ *p++ = (uint8_t)(v | 0x80); v >>= 7; }
    *p++ = (uint8_t)v; return p;
}
static uint8_t* write_svar32(uint8_t *p, int32_t v){
    return write_uvar32(p, zigzag32(v));
}
static inline int16_t q16(float x){
    if(x> 1.0f) x= 1.0f; if(x<-1.0f) x=-1.0f;
    int v = (int)lrintf(x * JLO2_SCALE);
    if(v<-32768) v=-32768; if(v>32767) v=32767;
    return (int16_t)v;
}
// ======================================================
// Config (match the Swift defaults)
// ======================================================
enum { N = 200000 };                 // samples
static const double FREQ = 13.0;     // waveform frequency
static const float  NOISE = 0.0f;    // optional noise
static const float  EPS_DEFAULT = 1e-4f;

// Epsilon sweep set
static const float EPSILONS[] = {1e-6f, 3e-6f, 1e-5f, 3e-5f, 1e-4f, 3e-4f, 1e-3f};
static const int   EPS_COUNT = sizeof(EPSILONS)/sizeof(EPSILONS[0]);

// ======================================================
// Timing (ms)
// ======================================================
static double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1000.0 + (double)ts.tv_nsec/1.0e6;
}

// ======================================================
// Signal (Float32)
// ======================================================
static void make_signal(float *x){
    for(int i=0;i<N;i++){
        double t = (double)i / (double)N;
        double v = sin(2.0*M_PI*FREQ*t)*0.7
                 + 0.25*sin(2.0*M_PI*(FREQ*0.37)*t + 0.6);
        // Optional noise
        if (NOISE != 0.0f){
            double r = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
            v += (double)NOISE * r;
        }
        x[i] = (float)v;
    }
}

// ======================================================
// J-L0 triangle compressor (piecewise-linear, ε-bounded)
// ======================================================
typedef struct { int i0; float v0; int i1; float v1; } Segment;

static inline float lerp(float a, float b, float t){ return a + (b-a)*t; }

static void maxDeviationIndex(const float *arr, int i0, float v0, int i1, float v1,
                              int *idx, float *err){
    int len = i1 - i0;
    if(len <= 1){ *idx=i0; *err=0.0f; return; }
    int worstIdx=i0; float worstErr=0.0f;
    float denom = (float)len;
    for(int i=i0+1;i<i1;i++){
        float t = (float)(i - i0) / denom;
        float p = lerp(v0, v1, t);
        float e = fabsf(arr[i] - p);
        if(e > worstErr){ worstErr = e; worstIdx = i; }
    }
    *idx = worstIdx; *err = worstErr;
}

// Simple recursive splitter (depth ~ O(#segments))
static Segment* compressJLO(const float *arr, int n, float eps, int *outCount){
    // Worst-case segments ~ 2*n in pathological cases; but typically far smaller.
    // We’ll accumulate in a dynamic array.
    int cap = 1024, count = 0;
    Segment *segs = (Segment*)malloc(sizeof(Segment)*cap);
    if(!segs){ fprintf(stderr,"OOM\n"); exit(1); }

    // Inner recursive lambda emulation
    struct Frame { int i0,i1; float v0,v1; int phase; int mid; float err; };
    // Convert recursion to an explicit stack to avoid deep recursion risks.
    struct Frame *stack = (struct Frame*)malloc(sizeof(struct Frame)* (n*2));
    int sp=0;
    stack[sp++] = (struct Frame){0, n-1, arr[0], arr[n-1], 0, 0, 0};

    while(sp>0){
        struct Frame f = stack[--sp];
        if(f.phase==0){
            int idx; float err;
            maxDeviationIndex(arr, f.i0, f.v0, f.i1, f.v1, &idx, &err);
            if(err <= eps || f.i1 - f.i0 <= 1){
                if(count==cap){ cap*=2; segs=(Segment*)realloc(segs,sizeof(Segment)*cap); }
                segs[count++] = (Segment){ f.i0, f.v0, f.i1, f.v1 };
                continue;
            }
            // split
            float vm = arr[idx];
            // Post-order push: right then left so left handled next
            stack[sp++] = (struct Frame){ idx, f.i1, vm, f.v1, 0, 0, 0 };
            stack[sp++] = (struct Frame){ f.i0, idx, f.v0, vm, 0, 0, 0 };
        }
    }
    free(stack);
    *outCount = count;
    return segs;
}

// ======================================================
// Reconstruct + error
// ======================================================
static void reconstruct(const Segment *segs, int segCount, int n, float *y){
    for(int i=0;i<n;i++) y[i]=0.0f;
    for(int s=0;s<segCount;s++){
        int i0=segs[s].i0, i1=segs[s].i1;
        float v0=segs[s].v0, v1=segs[s].v1;
        int len = i1 - i0;
        if(len <= 0){
            if(i0>=0 && i0<n) y[i0]=v0;
            continue;
        }
        for(int i=i0;i<=i1;i++){
            float t = (float)(i - i0) / (float)len;
            y[i] = lerp(v0, v1, t);
        }
    }
}

static void rms_and_max(const float *a, const float *b, int n, double *rms, double *maxabs){
    long double s=0.0L; double m=0.0;
    for(int i=0;i<n;i++){
        double e = (double)a[i] - (double)b[i];
        s += e*e; if (fabs(e)>m) m=fabs(e);
    }
    *rms = sqrt((double)(s / (long double)n)); *maxabs = m;
}

// ======================================================
// Save helpers
// ======================================================
static int saveJLO(const char *path, int n, float epsilon, const Segment *segs, int segCount){
    FILE *fp = fopen(path, "wb"); if(!fp) return 0;
    fwrite("JLO1", 1, 4, fp);
    int32_t n32=(int32_t)n, count=(int32_t)segCount;
    fwrite(&n32,   sizeof(int32_t), 1, fp);
    fwrite(&epsilon, sizeof(float), 1, fp);
    fwrite(&count, sizeof(int32_t), 1, fp);
    for(int i=0;i<segCount;i++){
        int32_t i0=segs[i].i0, i1=segs[i].i1;
        float   v0=segs[i].v0, v1=segs[i].v1;
        fwrite(&i0, sizeof(int32_t), 1, fp);
        fwrite(&v0, sizeof(float),   1, fp);
        fwrite(&i1, sizeof(int32_t), 1, fp);
        fwrite(&v1, sizeof(float),   1, fp);
    }
    fclose(fp); return 1;
}

static int saveCSV(const char *path, const float *arr, int n){
    FILE *fp=fopen(path,"wb"); if(!fp) return 0;
    char buf[64];
    for(int i=0;i<n;i++){
        int len = snprintf(buf,sizeof(buf),"%d,%.9g\n", i, arr[i]);
        fwrite(buf,1,(size_t)len,fp);
    }
    fclose(fp); return 1;
}

static int saveRawFloat32(const char *path, const float *arr, int n){
    FILE *fp=fopen(path,"wb"); if(!fp) return 0;
    fwrite(arr,sizeof(float),(size_t)n,fp);
    fclose(fp); return 1;
}

// ======================================================
// zlib compression helper (lossless baseline)
// ======================================================
static int compressFile_zlib(const char *inPath, const char *outPath){
    // Read input
    FILE *fi=fopen(inPath,"rb"); if(!fi) return 0;
    fseek(fi,0,SEEK_END); long len=ftell(fi); fseek(fi,0,SEEK_SET);
    uint8_t *src=(uint8_t*)malloc((size_t)len);
    if(!src){ fclose(fi); return 0; }
    fread(src,1,(size_t)len,fi); fclose(fi);

    uLongf bound = compressBound((uLong)len);
    uint8_t *dst=(uint8_t*)malloc(bound);
    if(!dst){ free(src); return 0; }

    uLongf outSize=bound;
    int rc = compress2(dst,&outSize,src,(uLong)len, Z_BEST_COMPRESSION);
    if(rc!=Z_OK){ free(src); free(dst); return 0; }

    FILE *fo=fopen(outPath,"wb");
    if(!fo){ free(src); free(dst); return 0; }
    fwrite(dst,1,(size_t)outSize,fo); fclose(fo);
    free(src); free(dst); return 1;
}

// ======================================================
// File size + formatting
// ======================================================
static unsigned long long fileSize(const char *path){
    FILE *fp=fopen(path,"rb"); if(!fp) return 0;
    fseek(fp,0,SEEK_END); long long s=ftell(fp); fclose(fp);
    return (unsigned long long)(s<0?0:s);
}
static void fmtBytes(unsigned long long b, char out[32]){
    if(b>10000000ULL) snprintf(out,32,"%.2f MB", (double)b/1e6);
    else if(b>10000ULL) snprintf(out,32,"%.2f KB", (double)b/1e3);
    else snprintf(out,32,"%llu B", b);
}

// ======================================================
// Simple JSON helpers (for HTML)
// ======================================================
static void write_json_array_d(FILE *fp, const double *a, int n){
    fputc('[',fp);
    for(int i=0;i<n;i++){ if(i) fputc(',',fp); fprintf(fp,"%.10g", a[i]); }
    fputc(']',fp);
}
static void write_json_array_i(FILE *fp, const int *a, int n){
    fputc('[',fp);
    for(int i=0;i<n;i++){ if(i) fputc(',',fp); fprintf(fp,"%d", a[i]); }
    fputc(']',fp);
}
static void write_json_array_s(FILE *fp, const char **s, int n){
    fputc('[',fp);
    for(int i=0;i<n;i++){ if(i) fputc(',',fp); fprintf(fp,"\"%s\"", s[i]); }
    fputc(']',fp);
}

// ======================================================
// Main
// ======================================================
int main(void){
    // 1) Generate signal
    float *x = (float*)malloc(sizeof(float)*N);
    float *y = (float*)malloc(sizeof(float)*N);
    if(!x||!y){ fprintf(stderr,"OOM\n"); return 1; }
    make_signal(x);

    // 2) Headline demo at EPS_DEFAULT
    double t0=now_ms();
    int segCount=0;
    Segment *segs = compressJLO(x, N, EPS_DEFAULT, &segCount);
    double t1=now_ms();

    reconstruct(segs, segCount, N, y);
    double t2=now_ms();

    double rms=0,maxA=0;
    rms_and_max(x,y,N,&rms,&maxA);

    // 3) Save outputs
    const char *jloPath="waveform.jlo";
    const char *csvPath="waveform.csv";
    const char *rawPath="waveform.rawf32";
    saveJLO(jloPath, N, EPS_DEFAULT, segs, segCount);
    saveCSV(csvPath, x, N);
    saveRawFloat32(rawPath, x, N);

    // 4) Lossless baselines (zlib)
    char csvZlib[64]="waveform.csv.zlib";
    char rawZlib[64]="waveform.rawf32.zlib";
    compressFile_zlib(csvPath, csvZlib);
    compressFile_zlib(rawPath, rawZlib);

    unsigned long long sizeJLO   = fileSize(jloPath);
    unsigned long long sizeCSV   = fileSize(csvPath);
    unsigned long long sizeRAW   = fileSize(rawPath);
    unsigned long long sizeCSVgz = fileSize(csvZlib);
    unsigned long long sizeRAWgz = fileSize(rawZlib);

    char bJ[32],bC[32],bR[32],bCg[32],bRg[32];
    fmtBytes(sizeJLO,bJ); fmtBytes(sizeCSV,bC); fmtBytes(sizeRAW,bR);
    fmtBytes(sizeCSVgz,bCg); fmtBytes(sizeRAWgz,bRg);

    printf("=== J-L0 Triangle Compression Demo ===\n");
    printf("Samples: %d | epsilon: %.1e\n", N, EPS_DEFAULT);
    printf("Segments: %d (avg span ~%.1f samples/seg)\n",
           segCount, (double)N/(double)segCount);
    printf("Compress time: %.3f s | Reconstruct time: %.3f s\n",
           (t1-t0)/1000.0, (t2-t1)/1000.0);
    printf("Error: RMS=%.3e | MaxAbs=%.3e\n", rms, maxA);

    printf("\nSize comparison (lossy JLO vs lossless zlib):\n");
    printf("JLO       : %s\n", bJ);
    printf("CSV       : %s\n", bC);
    printf("RAW f32   : %s\n", bR);
    printf("CSV+zlib  : %s\n", bCg);
    printf("RAW+zlib  : %s\n", bRg);
// ----------------------------------------------------------
// Compress the JLO payload itself (for storage benchmark)
// ----------------------------------------------------------
{
    // Rebuild header + segments in-memory
    size_t hdr_sz = 4 + sizeof(int32_t)*3; // "JLO1" + n32 + eps + count
    size_t seg_sz = (size_t)segCount * (sizeof(int32_t)*2 + sizeof(float)*2);
    size_t pay_sz = hdr_sz + seg_sz;

    uint8_t *payload = (uint8_t*)malloc(pay_sz);
    uint8_t *p = payload;

    memcpy(p, "JLO1", 4); p += 4;
    int32_t n32 = N, count32 = segCount;
    memcpy(p, &n32, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(p, &EPS_DEFAULT, sizeof(float)); p += sizeof(float);
    memcpy(p, &count32, sizeof(int32_t)); p += sizeof(int32_t);

    for (int i=0;i<segCount;i++){
        memcpy(p, &segs[i].i0, sizeof(int32_t)); p += sizeof(int32_t);
        memcpy(p, &segs[i].v0, sizeof(float));   p += sizeof(float);
        memcpy(p, &segs[i].i1, sizeof(int32_t)); p += sizeof(int32_t);
        memcpy(p, &segs[i].v1, sizeof(float));   p += sizeof(float);
    }

    uLongf bound = compressBound((uLong)pay_sz);
    uint8_t *cmp = (uint8_t*)malloc(bound);
    uLongf cmpLen = bound;

    double tC0 = now_ms();
    int rc = compress2(cmp, &cmpLen, payload, (uLong)pay_sz, Z_BEST_COMPRESSION);
    double tC1 = now_ms();

    if (rc == Z_OK) {
        printf("\nJLO payload (zlib-compressed): %.2f KB  (%.2fx smaller than raw CSV)  [%.2f ms]\n",
               (double)cmpLen/1024.0,
               (double)sizeCSV / (double)(cmpLen?cmpLen:1ULL),
               tC1 - tC0);
    } else {
        printf("\nJLO payload zlib compression failed (%d)\n", rc);
    }

    free(payload);
    free(cmp);
}
// ----------------------------------------------------------
// JLO2: compact varint+int16 packing of triangles (+zlib)
// Format:
//   "JL2\1" | N(u32) | eps(float32) | S(u32)
//   then for each segment:
//     span = i1 - i0 (uvar32), then int16 v0, int16 v1
//   (i0 is implied by summing spans; first i0 = 0)
// ----------------------------------------------------------
{
    // Worst-case buffer (very generous): header + 8 bytes/seg
    size_t cap = 16 + (size_t)segCount * 8u;
    uint8_t *buf = (uint8_t*)malloc(cap);
    uint8_t *p = buf;

    // header
    memcpy(p, "JL2\1", 4); p += 4;
    uint32_t N32 = (uint32_t)N, S32 = (uint32_t)segCount;
    memcpy(p, &N32, sizeof(uint32_t)); p += 4;
    memcpy(p, &EPS_DEFAULT, 4);        p += 4;
    memcpy(p, &S32, sizeof(uint32_t)); p += 4;

    // body
    int prev_i1 = 0;
    for(int s=0; s<segCount; ++s){
        int i0 = segs[s].i0, i1 = segs[s].i1;
        // enforce continuity; if your compressor guarantees non-overlap/ordered, this holds
        int span = i1 - i0;
        p = write_uvar32(p, (uint32_t)span);

        int16_t v0q = q16(segs[s].v0);
        int16_t v1q = q16(segs[s].v1);
        memcpy(p, &v0q, 2); p += 2;
        memcpy(p, &v1q, 2); p += 2;

        prev_i1 = i1;
    }

    size_t rawLen = (size_t)(p - buf);
    uLongf bound = compressBound((uLong)rawLen);
    uint8_t *cmp = (uint8_t*)malloc(bound);
    uLongf cmpLen = bound;

    double t0 = now_ms();
    int rc = compress2(cmp, &cmpLen, buf, (uLong)rawLen, Z_BEST_COMPRESSION);
    double t1 = now_ms();

    if (rc == Z_OK) {
        printf("JLO2 packed (varint+q16) raw: %.2f KB | zlib: %.2f KB  (%.2fx vs CSV)  [%.2f ms]\n",
               (double)rawLen/1024.0,
               (double)cmpLen/1024.0,
               (double)sizeCSV / (double)(cmpLen?cmpLen:1ULL),
               t1 - t0);
    } else {
        printf("JLO2 zlib compression failed (%d)\n", rc);
    }

    free(buf);
    free(cmp);
}
    printf("\nCompression ratio vs CSV: "
           "JLO=%.2fx, CSV+zlib=%.2fx, RAW f32=%.2fx, RAW+zlib=%.2fx\n",
           sizeCSV/(double)(sizeJLO?sizeJLO:1ULL),
           sizeCSV/(double)(sizeCSVgz?sizeCSVgz:1ULL),
           sizeCSV/(double)(sizeRAW?sizeRAW:1ULL),
           sizeCSV/(double)(sizeRAWgz?sizeRAWgz:1ULL));

    // 5) Rate–Distortion Sweep
    typedef struct {
        float epsilon;
        int segments;
        unsigned long long jloBytes;
        double rms, maxAbs, avgSpan;
    } Row;

    Row *rows = (Row*)malloc(sizeof(Row)*EPS_COUNT);
    if(!rows){ fprintf(stderr,"OOM\n"); return 1; }

    for(int k=0;k<EPS_COUNT;k++){
        float eps = EPSILONS[k];
        int sc=0;
        Segment *sg = compressJLO(x, N, eps, &sc);

        // measure JLO bytes in-memory
        unsigned long long jbytes = 4 + 4 + 4 + 4 + (unsigned long long)sc*(4+4+4+4);

        reconstruct(sg, sc, N, y);
        double r=0,mx=0; rms_and_max(x,y,N,&r,&mx);

        rows[k] = (Row){ eps, sc, jbytes, r, mx, (double)N/(double)(sc?sc:1) };
        free(sg);
    }

    // Print sweep
    printf("\n=== Rate–Distortion Sweep (ε → size/error) ===\n");
    printf("%-10s  %-9s  %-9s  %-10s  %-10s  %-10s\n",
           "epsilon","segments","avgSpan","JLO KB","RMS","MaxAbs");
    for(int k=0;k<EPS_COUNT;k++){
        double kb = (double)rows[k].jloBytes/1024.0;
        printf("%-10.1e  %-9d  %-9.1f  %-10.2f  %-10.3e  %-10.3e\n",
               rows[k].epsilon, rows[k].segments, rows[k].avgSpan,
               kb, rows[k].rms, rows[k].maxAbs);
    }

    // 6) Save sweep CSV
    const char *sweepCSV="jlo_sweep.csv";
    FILE *fcsv=fopen(sweepCSV,"wb");
    if(fcsv){
        fprintf(fcsv,"epsilon,segments,avgSpan,jlo_bytes,jlo_kb,rms,max_abs\n");
        for(int k=0;k<EPS_COUNT;k++){
            fprintf(fcsv,"%.6g,%d,%.3f,%llu,%.2f,%.8e,%.8e\n",
                rows[k].epsilon, rows[k].segments, rows[k].avgSpan,
                rows[k].jloBytes, (double)rows[k].jloBytes/1024.0,
                rows[k].rms, rows[k].maxAbs);
        }
        fclose(fcsv);
        printf("\nSweep CSV written to: %s\n", sweepCSV);
    }else{
        printf("\nFailed to write sweep CSV.\n");
    }

    // 7) HTML dashboard (Chart.js)
    const char *htmlPath="jlo_sweep.html";
    FILE *fh=fopen(htmlPath,"wb");
    if(fh){
        // Prepare arrays
        double sizeKB[EPS_COUNT], rmsA[EPS_COUNT], maxAarr[EPS_COUNT], spanA[EPS_COUNT];
        int segArr[EPS_COUNT];
        const char *epsLabel[EPS_COUNT];
        char epsBuf[EPS_COUNT][32];
        for(int k=0;k<EPS_COUNT;k++){
            sizeKB[k] = (double)rows[k].jloBytes/1024.0;
            rmsA[k] = rows[k].rms; maxAarr[k] = rows[k].maxAbs;
            spanA[k] = rows[k].avgSpan; segArr[k] = rows[k].segments;
            snprintf(epsBuf[k],32,"%.8g", rows[k].epsilon);
            epsLabel[k] = epsBuf[k];
        }

        // HTML/JS
        fprintf(fh,
"<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n"
"<title>J-L0 Rate–Distortion Sweep</title>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
"<style>body{font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;} .wrap{max-width:1000px;margin:20px auto;padding:0 12px;} h1{font-size:20px;margin:12px 0;} canvas{max-width:100%%;height:360px;} .grid{display:grid;grid-template-columns:1fr;gap:24px} @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}} .card{border:1px solid #eee;border-radius:10px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,0.04);} code{background:#f7f7f7;padding:2px 6px;border-radius:6px}</style>\n"
"<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n</head>\n<body>\n<div class=\"wrap\">\n<h1>J-L0 Rate–Distortion Sweep</h1>\n");

        fprintf(fh,"<p>Epsilons: <code>");
        for(int k=0;k<EPS_COUNT;k++){ if(k) fputs(", ",fh); fputs(epsLabel[k],fh); }
        fprintf(fh,"</code></p>\n<div class=\"grid\">\n"
                   "<div class=\"card\"><h3>JLO size (KB) vs ε</h3><canvas id=\"sizeChart\"></canvas></div>\n"
                   "<div class=\"card\"><h3>Segments & Avg Span vs ε</h3><canvas id=\"segChart\"></canvas></div>\n"
                   "<div class=\"card\"><h3>RMS error vs ε</h3><canvas id=\"rmsChart\"></canvas></div>\n"
                   "<div class=\"card\"><h3>MaxAbs error vs ε</h3><canvas id=\"maxChart\"></canvas></div>\n"
                   "</div>\n</div>\n<script>\nconst labels = ");
        write_json_array_s(fh, epsLabel, EPS_COUNT); fputs(";\nconst sizeKB = ", fh);
        write_json_array_d(fh, sizeKB, EPS_COUNT); fputs(";\nconst segments = ", fh);
        write_json_array_i(fh, segArr, EPS_COUNT); fputs(";\nconst avgSpan = ", fh);
        write_json_array_d(fh, spanA, EPS_COUNT); fputs(";\nconst rms = ", fh);
        write_json_array_d(fh, rmsA, EPS_COUNT); fputs(";\nconst maxAbs = ", fh);
        write_json_array_d(fh, maxAarr, EPS_COUNT); fputs(";\n", fh);

        // JS chart helper + charts
        fputs(
"function mkChart(id, label, data, extraDataset=null, ylog=false){\n"
"  const ctx=document.getElementById(id);\n"
"  new Chart(ctx,{type:'line',data:{labels:labels,datasets:[{label:label,data:data,tension:0.2},...(extraDataset?[extraDataset]:[])]},options:{responsive:true,scales:{y:{type:ylog?'logarithmic':'linear'},x:{ticks:{maxRotation:0,autoSkip:true}}},plugins:{legend:{display:true}}});\n"
"}\n"
"mkChart('sizeChart','JLO size (KB)', sizeKB);\n"
"mkChart('segChart','segments', segments, {label:'avgSpan', data: avgSpan, tension:0.2});\n"
"mkChart('rmsChart','RMS error', rms, null, true);\n"
"mkChart('maxChart','MaxAbs error', maxAbs, null, true);\n"
"</script>\n</body>\n</html>\n", fh);

        fclose(fh);
        printf("HTML dashboard written to: %s\n", htmlPath);
        printf("Open it in a browser to see the graphs.\n");
    }else{
        printf("Failed to write HTML dashboard.\n");
    }

    free(rows); free(segs); free(x); free(y);
    return 0;
}
#include <metal_stdlib>
using namespace metal;

#define TERMS 28u
#define GROUP_SIZE 32u
#define FRAC_BITS 40
#define SCALE_F 1099511627776.0f // 2^40

struct LogDebugOut {
    float approx;
    float exact;
    float absErr;
    uint termCount;
};

/*
    log2 additive correction terms for factors:
      f_k = 1 + 2^-k, k=1..28

    Stored as float for factor tests and signed fixed-point for accumulation.
    Fixed-point scale: 2^40.
*/

constant float FACTORS[TERMS] = {
    1.5f,
    1.25f,
    1.125f,
    1.0625f,
    1.03125f,
    1.015625f,
    1.0078125f,
    1.00390625f,
    1.001953125f,
    1.0009765625f,
    1.00048828125f,
    1.000244140625f,
    1.0001220703125f,
    1.00006103515625f,
    1.000030517578125f,
    1.0000152587890625f,
    1.00000762939453125f,
    1.000003814697265625f,
    1.0000019073486328125f,
    1.00000095367431640625f,
    1.000000476837158203125f,
    1.0000002384185791015625f,
    1.00000011920928955078125f,
    1.000000059604644775390625f,
    1.0000000298023223876953125f,
    1.00000001490116119384765625f,
    1.000000007450580596923828125f,
    1.0000000037252902984619140625f
};

constant long LOG_TERMS_Q40[TERMS] = {
    643057143072L,  // log2(1.5)
    354636633008L,  // log2(1.25)
    184424571137L,
    94089242358L,
    47506268250L,
    23872112548L,
    11965987740L,
    5990527982L,
    2996760415L,
    1498754684L,
    749470944L,
    374758280L,
    187384842L,
    93693847L,
    46847066L,
    23423561L,
    11711787L,
    5855895L,
    2927948L,
    1463974L,
    731987L,
    365993L,
    182997L,
    91498L,
    45749L,
    22875L,
    11437L,
    5719L
};

inline ulong token_add_u64(ulong a, ulong b) {
    for (uint i = 0u; i < 128u && b != 0ul; i++) {
        ulong carry = (a & b) << 1u;
        a = a ^ b;
        b = carry;
    }
    return a;
}

inline long signed_q40_from_float(float x) {
    return (long)rint(x * SCALE_F);
}

inline float q40_to_float(long q) {
    return ((float)q) / SCALE_F;
}

/*
    Serial per-value additive log.
    One GPU thread handles one x.
*/
kernel void log2AddSerialKernel(
    device const float *X [[buffer(0)]],
    device float *OUT [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;

    float x = X[id];

    int exponent = 0;
    float m = frexp(x, exponent); // m in [0.5,1)
    m *= 2.0f;
    exponent -= 1;

    long acc = ((long)exponent) << FRAC_BITS;

    for (uint k = 0u; k < TERMS; k++) {
        float f = FACTORS[k];

        if (m >= f) {
            m /= f;
            acc += LOG_TERMS_Q40[k];
        }
    }

    OUT[id] = q40_to_float(acc);
}

/*
    Native Metal log2 baseline.
*/
kernel void log2NativeKernel(
    device const float *X [[buffer(0)]],
    device float *OUT [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    OUT[id] = log2(X[id]);
}

/*
    Parallel lane version:
      one threadgroup handles one x
      lanes 0..27 each test/generate one correction term

    Important limitation:
      greedy factor decisions are sequential because m changes after each accepted term.
      This parallel version uses a prefix-style approximation:
        each lane tests against the original normalized m.
      This is NOT the same as exact greedy decomposition.
      It tests whether massively parallel term generation + reduction is cheap/useful.

    Output is approximate and expected to be less accurate than serial greedy.
*/
kernel void log2AddParallelTermsKernel(
    device const float *X [[buffer(0)]],
    device float *OUT [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint groupID [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (groupID >= N || lane >= GROUP_SIZE) return;

    threadgroup ulong terms[GROUP_SIZE];

    float x = X[groupID];

    int exponent = 0;
    float m = frexp(x, exponent);
    m *= 2.0f;
    exponent -= 1;

    ulong term = 0ul;

    if (lane == 0u) {
        term = (ulong)(((long)exponent) << FRAC_BITS);
    } else if (lane - 1u < TERMS) {
        uint k = lane - 1u;
        float f = FACTORS[k];

        /*
            Non-greedy independent term decision.
            Cheaper/parallel, but less mathematically exact.
        */
        if (m >= f) {
            term = (ulong)LOG_TERMS_Q40[k];
        }
    }

    terms[lane] = term;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /*
        Parallel XOR/carry-token reduction tree.
        This is intentionally using token_add to mimic shift-carry recombination.
    */
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            terms[lane] = token_add_u64(terms[lane], terms[lane + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        long signedResult = (long)terms[0];
        OUT[groupID] = q40_to_float(signedResult);
    }
}

/*
    Debug version for accuracy comparison.
*/
kernel void log2AddParallelDebugKernel(
    device const float *X [[buffer(0)]],
    device LogDebugOut *OUT [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint groupID [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (groupID >= N || lane >= GROUP_SIZE) return;

    threadgroup ulong terms[GROUP_SIZE];
    threadgroup uint accepted[GROUP_SIZE];

    float x = X[groupID];

    int exponent = 0;
    float m = frexp(x, exponent);
    m *= 2.0f;
    exponent -= 1;

    ulong term = 0ul;
    uint accTerm = 0u;

    if (lane == 0u) {
        term = (ulong)(((long)exponent) << FRAC_BITS);
    } else if (lane - 1u < TERMS) {
        uint k = lane - 1u;
        if (m >= FACTORS[k]) {
            term = (ulong)LOG_TERMS_Q40[k];
            accTerm = 1u;
        }
    }

    terms[lane] = term;
    accepted[lane] = accTerm;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            terms[lane] = token_add_u64(terms[lane], terms[lane + stride]);
            accepted[lane] += accepted[lane + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        long signedResult = (long)terms[0];
        float approx = q40_to_float(signedResult);
        float exact = log2(x);

        LogDebugOut d;
        d.approx = approx;
        d.exact = exact;
        d.absErr = fabs(approx - exact);
        d.termCount = accepted[0];

        OUT[groupID] = d;
    }
}

#include <metal_stdlib>
using namespace metal;

#define B4_WIDTH64 32u

struct DivOut {
    ulong q;
    ulong r;
};

struct TokenDivOut {
    ulong q;
    ulong r;
    uint waves;
};

struct SubOut {
    ulong diff;
    uint borrow;
    uint waves;
};

inline uint b4_digit64(ulong x, uint pos) {
    return (uint)((x >> (pos * 2u)) & 3ul);
}

inline SubOut sub64_borrow_token(ulong a, ulong b) {
    ulong low = 0ul;
    ulong borrowStream = 0ul;
    uint borrowOut = 0u;

    for (uint pos = 0u; pos < B4_WIDTH64; pos++) {
        uint ad = b4_digit64(a, pos);
        uint bd = b4_digit64(b, pos);

        uint outDigit;
        if (ad >= bd) {
            outDigit = ad - bd;
        } else {
            outDigit = ad + 4u - bd;
            if (pos < B4_WIDTH64 - 1u) {
                borrowStream |= 1ul << ((pos + 1u) * 2u);
            } else {
                borrowOut = 1u;
            }
        }

        low |= ((ulong)outDigit) << (pos * 2u);
    }

    uint waves = 0u;

    for (uint wave = 0u; wave < B4_WIDTH64 && borrowStream != 0ul; wave++) {
        waves++;

        ulong nextLow = 0ul;
        ulong nextBorrow = 0ul;

        for (uint pos = 0u; pos < B4_WIDTH64; pos++) {
            uint ld = b4_digit64(low, pos);
            uint tok = b4_digit64(borrowStream, pos);

            uint outDigit;
            if (ld >= tok) {
                outDigit = ld - tok;
            } else {
                outDigit = ld + 4u - tok;
                if (pos < B4_WIDTH64 - 1u) {
                    nextBorrow |= 1ul << ((pos + 1u) * 2u);
                } else {
                    borrowOut = 1u;
                }
            }

            nextLow |= ((ulong)outDigit) << (pos * 2u);
        }

        low = nextLow;
        borrowStream = nextBorrow;
    }

    SubOut s;
    s.diff = low;
    s.borrow = borrowOut;
    s.waves = waves;
    return s;
}

/*
    Original branch-light radix-4 divider.
*/
inline void div64_radix4_original_core(ulong x, ulong y, thread ulong& q, thread ulong& r)
{
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;

    ulong t = (k ? (x >> k) : x);
    ulong w = (k ? (x & ((1ul << k) - 1ul)) : 0ul);

    if (y1 == 1ul) {
        q = t;
        r = (k ? w : 0ul);
        return;
    }

    ulong Q = 0ul;
    ulong R = 0ul;
    ulong yv = y1;

    uint msb = (t == 0ul) ? 0u : (63u - (uint)clz(t));
    int pairs = (int)(msb / 2u);

    for (int j = pairs; j >= 0; --j) {
        uint i = (uint)(j * 2u);
        R = (R << 2) | ((t >> i) & 3ul);

        ulong ge1 = (ulong)(R >= yv);
        ulong R1  = R - (yv & -(long)ge1);

        ulong ge2 = (ulong)(R1 >= yv);
        ulong R2  = R1 - (yv & -(long)ge2);

        ulong ge3 = (ulong)(R2 >= yv);
        ulong R3  = R2 - (yv & -(long)ge3);

        ulong addQ = ge1 + ge2 + ge3;
        R = R3;
        Q |= (addQ << i);
    }

    ulong r0 = R;
    ulong r_orig = (k ? ((r0 << k) + w) : r0);
    q = Q;
    r = r_orig;
}

/*
    Token-sub divider with radix-4 normalization:
      Shift the odd divisor y1 left by an EVEN bit count so its MSB lands
      on a base-4 digit boundary. Shift t by the same amount into the
      normalized domain where trial subtracts happen.

    After quotient is found, remainder is shifted back down.

    This is experimental. Correctness tells us if the mapping survives.
*/
inline void div64_token_norm_core(ulong x, ulong y, thread ulong& q, thread ulong& r, thread uint& waveCount)
{
    uint k = (uint)ctz(y);
    ulong y1 = y >> k;

    ulong t = (k ? (x >> k) : x);
    ulong w = (k ? (x & ((1ul << k) - 1ul)) : 0ul);

    waveCount = 0u;

    if (y1 == 1ul) {
        q = t;
        r = (k ? w : 0ul);
        return;
    }

    /*
        Choose a base-4-aligned normalization shift.
        If y1's highest bit is at an odd bit position, shift by 1 to align.
        But keep the shift small and safe.
    */
    uint ymsb = 63u - (uint)clz(y1);
    uint normShift = (ymsb & 1u) ? 1u : 0u;

    ulong yv = y1 << normShift;
    ulong tn = t << normShift;

    ulong Q = 0ul;
    ulong R = 0ul;

    uint msb = (tn == 0ul) ? 0u : (63u - (uint)clz(tn));
    int pairs = (int)(msb / 2u);

    for (int j = pairs; j >= 0; --j) {
        uint i = (uint)(j * 2u);
        R = (R << 2) | ((tn >> i) & 3ul);

        SubOut s1 = sub64_borrow_token(R, yv);
        waveCount += s1.waves;
        uint ge1 = 1u - s1.borrow;
        ulong R1 = ge1 ? s1.diff : R;

        SubOut s2 = sub64_borrow_token(R1, yv);
        waveCount += s2.waves;
        uint ge2 = 1u - s2.borrow;
        ulong R2 = ge2 ? s2.diff : R1;

        SubOut s3 = sub64_borrow_token(R2, yv);
        waveCount += s3.waves;
        uint ge3 = 1u - s3.borrow;
        ulong R3 = ge3 ? s3.diff : R2;

        ulong addQ = (ulong)(ge1 + ge2 + ge3);
        R = R3;
        Q |= (addQ << i);
    }

    /*
        Normalize quotient back if tn had one extra low zero bit.
        Since we shifted dividend and divisor equally, quotient should match.
        R is in normalized remainder domain; shift back.
    */
    ulong r0 = (normShift ? (R >> normShift) : R);
    ulong r_orig = (k ? ((r0 << k) + w) : r0);

    q = Q;
    r = r_orig;
}

/*
    Token-sub divider where we precondition the divisor by stripping base-4 zero
    digits, not just bit zeros. This makes divisor odd part also base-4 compact.
*/
inline void div64_token_base4compact_core(ulong x, ulong y, thread ulong& q, thread ulong& r, thread uint& waveCount)
{
    /*
        Strip powers of 4 first where possible, then normal v7 even-first handles the rest.
        y = 4^m * y4
    */
    uint k2 = 0u;
    ulong yy = y;

    while ((yy & 3ul) == 0ul && yy != 0ul && k2 < 31u) {
        yy >>= 2u;
        k2 += 2u;
    }

    ulong t = x >> k2;
    ulong w = (k2 ? (x & ((1ul << k2) - 1ul)) : 0ul);

    ulong q0, r0;
    uint waves0;
    div64_token_norm_core(t, yy, q0, r0, waves0);

    waveCount = waves0;
    q = q0;
    r = (r0 << k2) + w;
}

kernel void div64OriginalKernel(
    device const ulong *X [[buffer(0)]],
    device const ulong *Y [[buffer(1)]],
    device DivOut *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    ulong q, r;
    div64_radix4_original_core(X[id], Y[id], q, r);
    OUT[id].q = q;
    OUT[id].r = r;
}

kernel void div64TokenNormKernel(
    device const ulong *X [[buffer(0)]],
    device const ulong *Y [[buffer(1)]],
    device TokenDivOut *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    ulong q, r;
    uint waves;
    div64_token_norm_core(X[id], Y[id], q, r, waves);
    OUT[id].q = q;
    OUT[id].r = r;
    OUT[id].waves = waves;
}

kernel void div64TokenBase4CompactKernel(
    device const ulong *X [[buffer(0)]],
    device const ulong *Y [[buffer(1)]],
    device TokenDivOut *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    ulong q, r;
    uint waves;
    div64_token_base4compact_core(X[id], Y[id], q, r, waves);
    OUT[id].q = q;
    OUT[id].r = r;
    OUT[id].waves = waves;
}

kernel void sub64TokenKernel(
    device const ulong *A [[buffer(0)]],
    device const ulong *B [[buffer(1)]],
    device TokenDivOut *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    SubOut s = sub64_borrow_token(A[id], B[id]);
    OUT[id].q = s.diff;
    OUT[id].r = (ulong)s.borrow;
    OUT[id].waves = s.waves;
}

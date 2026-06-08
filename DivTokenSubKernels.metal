#include <metal_stdlib>
using namespace metal;

#define B4_WIDTH64 32u

struct DivOut {
    ulong q;
    ulong r;
};

struct SubOut {
    ulong diff;
    uint borrow;
    uint waves;
};

/*
    Base-4 borrow-token subtract:
      computes a - b modulo 64 bits.
      borrow=1 means a < b.
      Borrow tokens move upward one base-4 digit at a time.

    This is the subtraction analogue of the carry-token idea:
      local digit subtract
      emit shifted borrow tokens
      resolve borrow waves
*/
inline uint b4_digit64(ulong x, uint pos) {
    return (uint)((x >> (pos * 2u)) & 3ul);
}

inline void set_b4_digit64(thread ulong &x, uint pos, uint digit) {
    ulong shift = (ulong)(pos * 2u);
    x &= ~(3ul << shift);
    x |= ((ulong)(digit & 3u)) << shift;
}

inline SubOut sub64_borrow_token(ulong a, ulong b) {
    ulong low = 0ul;
    ulong borrowStream = 0ul;

    uint initialBorrows = 0u;

    // local digit subtract without incoming borrow
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
                initialBorrows = 1u;
            }
        }

        low |= ((ulong)outDigit) << (pos * 2u);
    }

    uint borrowOut = initialBorrows;
    uint waves = 0u;

    // Resolve borrow tokens
    for (uint wave = 0u; wave < B4_WIDTH64 && borrowStream != 0ul; wave++) {
        waves++;

        ulong nextLow = 0ul;
        ulong nextBorrow = 0ul;

        for (uint pos = 0u; pos < B4_WIDTH64; pos++) {
            uint ld = b4_digit64(low, pos);
            uint tok = b4_digit64(borrowStream, pos); // normally 0 or 1

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
    Original v7 divider core.
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
    Token-subtractor divider core.

    Instead of comparison + native subtract, do:
      trial = token_sub(R, yv)
      if no borrow, accept.
    Repeat up to three times for radix-4 quotient digit.
*/
inline void div64_radix4_token_sub_core(ulong x, ulong y, thread ulong& q, thread ulong& r, thread uint& waveCount)
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

    ulong Q = 0ul;
    ulong R = 0ul;
    ulong yv = y1;

    uint msb = (t == 0ul) ? 0u : (63u - (uint)clz(t));
    int pairs = (int)(msb / 2u);

    for (int j = pairs; j >= 0; --j) {
        uint i = (uint)(j * 2u);
        R = (R << 2) | ((t >> i) & 3ul);

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

    ulong r0 = R;
    ulong r_orig = (k ? ((r0 << k) + w) : r0);
    q = Q;
    r = r_orig;
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

struct TokenDivOut {
    ulong q;
    ulong r;
    uint waves;
};

kernel void div64TokenSubKernel(
    device const ulong *X [[buffer(0)]],
    device const ulong *Y [[buffer(1)]],
    device TokenDivOut *OUT [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    ulong q, r;
    uint waves;
    div64_radix4_token_sub_core(X[id], Y[id], q, r, waves);
    OUT[id].q = q;
    OUT[id].r = r;
    OUT[id].waves = waves;
}

/*
    Standalone subtract microbench, useful for checking token subtract itself.
*/
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

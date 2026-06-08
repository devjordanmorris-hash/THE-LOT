#include <metal_stdlib>
using namespace metal;

ulong mul_bitwise_u64(ulong a, ulong b) {
    ulong res = 0;
    while (b) {
        if (b & 1) res += a;
        a <<= 1;  b >>= 1;
    }
    return res;
}

kernel void mul_array(device const ulong* A [[buffer(0)]],
                      device const ulong* B [[buffer(1)]],
                      device       ulong* C [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    C[gid] = mul_bitwise_u64(A[gid], B[gid]);
}

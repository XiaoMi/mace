#include <common.h>

// C = A * B
__kernel void gemm(__read_only image2d_t A,
                   __read_only image2d_t B,
                   __write_only image2d_t C,
                   __private const int M,
                   __private const int height_blocks,
                   __private const int K) {
  const int gx = get_global_id(0);
  const int hb = get_global_id(1);
  const int batch = hb / height_blocks;
  const int gy = (hb % height_blocks) << 2;
  const int bm = mul24(batch, M);
  const int bk = mul24(batch, K);

  float4 a0, a1, a2, a3;
  float4 b0, b1, b2, b3;
  float4 c0, c1, c2, c3;

  for (short pos = 0; pos < K; pos += 4) {
    a0 = READ_IMAGET(A, SAMPLER, (int2)(pos >> 2, (bm + gy)));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(pos >> 2, (bm + gy + 1)));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(pos >> 2, (bm + gy + 2)));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(pos >> 2, (bm + gy + 3)));

    b0 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos)));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos + 1)));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos + 2)));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos + 3)));

    c0 = mad(a0.x, b0, c0);
    c0 = mad(a0.y, b1, c0);
    c0 = mad(a0.z, b2, c0);
    c0 = mad(a0.w, b3, c0);

    c1 = mad(a1.x, b0, c1);
    c1 = mad(a1.y, b1, c1);
    c1 = mad(a1.z, b2, c1);
    c1 = mad(a1.w, b3, c1);

    c2 = mad(a2.x, b0, c2);
    c2 = mad(a2.y, b1, c2);
    c2 = mad(a2.z, b2, c2);
    c2 = mad(a2.w, b3, c2);

    c3 = mad(a3.x, b0, c3);
    c3 = mad(a3.y, b1, c3);
    c3 = mad(a3.z, b2, c3);
    c3 = mad(a3.w, b3, c3);
  }
  if (gy >= M) return;
  WRITE_IMAGET(C, (int2)(gx, (bm + gy)), c0);
  if ((gy + 1) >= M) return;
  WRITE_IMAGET(C, (int2)(gx, (bm + gy + 1)), c1);
  if ((gy + 2) >= M) return;
  WRITE_IMAGET(C, (int2)(gx, (bm + gy + 2)), c2);
  if ((gy + 3) >= M) return;
  WRITE_IMAGET(C, (int2)(gx, (bm + gy + 3)), c3);
}

#include <common.h>

// C = A * B
__kernel void matmul(OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM2
                     __read_only image2d_t A,
                     __read_only image2d_t B,
                     __write_only image2d_t C,
                     __private const int M,
                     __private const int N,
                     __private const int K,
                     __private const int height_blocks,
                     __private const int k_blocks) {
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int bm = mad24(batch, M, ty << 2);
  const int bk = mul24(batch, k_blocks);

  float4 a0, a1, a2, a3;
  float4 b0, b1, b2, b3;
  float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

  for (short pos = 0; pos < k_blocks; pos += 1) {
    a0 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm)));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 1)));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 2)));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 3)));

    b0 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos)));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(gx + 1, (bk + pos)));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(gx + 2, (bk + pos)));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(gx + 3, (bk + pos)));

    c0 += (DATA_TYPE4)(dot(a0, b0), dot(a1, b0), dot(a2, b0), dot(a3, b0));

    c1 += (DATA_TYPE4)(dot(a0, b1), dot(a1, b1), dot(a2, b1), dot(a3, b1));

    c2 += (DATA_TYPE4)(dot(a0, b2), dot(a1, b2), dot(a2, b2), dot(a3, b2));

    c3 += (DATA_TYPE4)(dot(a0, b3), dot(a1, b3), dot(a2, b3), dot(a3, b3));
  }

  WRITE_IMAGET(C, (int2)(gx, gy), c0);

  if ((gx + 1) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 1, gy), c1);

  if ((gx + 2) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 2, gy), c2);

  if ((gx + 3) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 3, gy), c3);
}

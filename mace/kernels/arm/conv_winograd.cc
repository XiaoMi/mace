// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <algorithm>

#include "mace/kernels/arm/conv_winograd.h"
#include "mace/kernels/gemm.h"
#include "mace/utils/logging.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

namespace {
// NCHW => NTCB (T: in tile pixels, B: tile indices)
void TransformInput4x4(const float *input,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t tile_count,
                       float *output) {
  const index_t stride = in_channels * tile_count;
  const index_t in_height_width = in_height * in_width;
  const index_t input_batch_size = in_height_width * in_channels;
  const index_t output_batch_size = 16 * in_channels * tile_count;

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t c = 0; c < in_channels; ++c) {
      index_t tile_index = 0;
      for (index_t h = 0; h < in_height - 2; h += 2) {
        for (index_t w = 0; w < in_width - 2; w += 2) {
          float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14,
              d15;
          float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15;

          // load tile data
          const float *input_ptr = input + n * input_batch_size +
                                   c * in_height_width + h * in_width + w;
          d0 = input_ptr[0];
          d1 = input_ptr[1];
          d2 = input_ptr[2];
          d3 = input_ptr[3];

          d4 = input_ptr[in_width];
          d5 = input_ptr[in_width + 1];
          d6 = input_ptr[in_width + 2];
          d7 = input_ptr[in_width + 3];

          d8 = input_ptr[2 * in_width];
          d9 = input_ptr[2 * in_width + 1];
          d10 = input_ptr[2 * in_width + 2];
          d11 = input_ptr[2 * in_width + 3];

          d12 = input_ptr[3 * in_width];
          d13 = input_ptr[3 * in_width + 1];
          d14 = input_ptr[3 * in_width + 2];
          d15 = input_ptr[3 * in_width + 3];

          // s = BT * d * B
          s0 = (d0 - d8) - (d2 - d10);
          s1 = (d1 - d9) + (d2 - d10);
          s2 = (d2 - d10) - (d1 - d9);
          s3 = (d1 - d9) - (d3 - d11);
          s4 = (d4 + d8) - (d6 + d10);
          s5 = (d5 + d9) + (d6 + d10);
          s6 = (d6 + d10) - (d5 + d9);
          s7 = (d5 + d9) - (d7 + d11);
          s8 = (d8 - d4) - (d10 - d6);
          s9 = (d9 - d5) + (d10 - d6);
          s10 = (d10 - d6) - (d9 - d5);
          s11 = (d9 - d5) - (d11 - d7);
          s12 = (d4 - d12) - (d6 - d14);
          s13 = (d5 - d13) + (d6 - d14);
          s14 = (d6 - d14) - (d5 - d13);
          s15 = (d5 - d13) - (d7 - d15);

          // store output
          float *output_ptr =
              output + n * output_batch_size + c * tile_count + tile_index;
          output_ptr[0] = s0;
          output_ptr[1 * stride] = s1;
          output_ptr[2 * stride] = s2;
          output_ptr[3 * stride] = s3;

          output_ptr[4 * stride] = s4;
          output_ptr[5 * stride] = s5;
          output_ptr[6 * stride] = s6;
          output_ptr[7 * stride] = s7;

          output_ptr[8 * stride] = s8;
          output_ptr[9 * stride] = s9;
          output_ptr[10 * stride] = s10;
          output_ptr[11 * stride] = s11;

          output_ptr[12 * stride] = s12;
          output_ptr[13 * stride] = s13;
          output_ptr[14 * stride] = s14;
          output_ptr[15 * stride] = s15;

          ++tile_index;
        }
      }
    }
  }
}

// NCHW => NTCB (T: in tile pixels, B: tile indices)
/**
 * BT =
⎡1   0    -21/4    0    21/4     0    -1  0⎤
⎢                                          ⎥
⎢0   1      1    -17/4  -17/4    1    1   0⎥
⎢                                          ⎥
⎢0   -1     1    17/4   -17/4   -1    1   0⎥
⎢                                          ⎥
⎢0  1/2    1/4   -5/2   -5/4     2    1   0⎥
⎢                                          ⎥
⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
⎢                                          ⎥
⎢0   2      4    -5/2    -5     1/2   1   0⎥
⎢                                          ⎥
⎢0   -2     4     5/2    -5    -1/2   1   0⎥
⎢                                          ⎥
⎣0   -1     0    21/4     0    -21/4  0   1⎦

 * @param input
 * @param batch
 * @param in_height
 * @param in_width
 * @param in_channels
 * @param tile_count
 * @param output
 */
void TransformInput8x8(const float *input,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t tile_count,
                       float *output) {
  const index_t stride = in_channels * tile_count;
  const index_t in_height_width = in_height * in_width;
  const index_t input_batch_size = in_height_width * in_channels;
  const index_t output_batch_size = 64 * in_channels * tile_count;

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t c = 0; c < in_channels; ++c) {
      index_t tile_index = 0;
      float s[8][8];
      for (index_t h = 0; h < in_height - 2; h += 6) {
        for (index_t w = 0; w < in_width - 2; w += 6) {
          const float *input_ptr = input + n * input_batch_size +
                                   c * in_height_width + h * in_width + w;

          for (int i = 0; i < 8; ++i) {
            float d0, d1, d2, d3, d4, d5, d6, d7;
            d0 = input_ptr[0];
            d1 = input_ptr[1];
            d2 = input_ptr[2];
            d3 = input_ptr[3];
            d4 = input_ptr[4];
            d5 = input_ptr[5];
            d6 = input_ptr[6];
            d7 = input_ptr[7];

            s[i][0] = d0 - d6 + (d4 - d2) * 5.25;
            s[i][7] = d7 - d1 + (d3 - d5) * 5.25;

            float u = d2 + d6 - d4 * 4.25;
            float v = d1 + d5 - d3 * 4.25;
            s[i][1] = u + v;
            s[i][2] = u - v;

            u = d6 + d2 * 0.25 - d4 * 1.25;
            v = d1 * 0.5 - d3 * 2.5 + d5 * 2;
            s[i][3] = u + v;
            s[i][4] = u - v;

            u = d6 + (d2 - d4 * 1.25) * 4;
            v = d1 * 2 - d3 * 2.5 + d5 * 0.5;
            s[i][5] = u + v;
            s[i][6] = u - v;

            input_ptr += in_width;
          }

          float *output_ptr =
              output + n * output_batch_size + c * tile_count + tile_index;
          for (int i = 0; i < 8; ++i) {
            float d0, d1, d2, d3, d4, d5, d6, d7;
            d0 = s[0][i];
            d1 = s[1][i];
            d2 = s[2][i];
            d3 = s[3][i];
            d4 = s[4][i];
            d5 = s[5][i];
            d6 = s[6][i];
            d7 = s[7][i];

            output_ptr[i * stride] = d0 - d6 + (d4 - d2) * 5.25;
            output_ptr[(56 + i) * stride] = d7 - d1 + (d3 - d5) * 5.25;

            float u = d2 + d6 - d4 * 4.25;
            float v = d1 + d5 - d3 * 4.25;
            output_ptr[(8 + i) * stride] = u + v;
            output_ptr[(16 + i) * stride] = u - v;

            u = d6 + d2 * 0.25 - d4 * 1.25;
            v = d1 * 0.5 - d3 * 2.5 + d5 * 2;
            output_ptr[(24 + i) * stride] = u + v;
            output_ptr[(32 + i) * stride] = u - v;

            u = d6 + (d2 - d4 * 1.25) * 4;
            v = d1 * 2 - d3 * 2.5 + d5 * 0.5;
            output_ptr[(40 + i) * stride] = u + v;
            output_ptr[(48 + i) * stride] = u - v;
          }

          ++tile_index;
        }
      }
    }
  }
}

// TOC * NTCB => NTOB
void BatchGemm(const float *input,
               const float *filter,
               index_t batch,
               index_t in_channels,
               index_t out_channels,
               index_t tile_count,
               int out_tile_size,
               float *output) {
  const index_t filter_stride = out_channels * in_channels;
  const int in_tile_area = (out_tile_size + 2) * (out_tile_size + 2);
  const index_t in_batch_size = in_tile_area * in_channels * tile_count;
  const index_t in_stride = in_channels * tile_count;
  const index_t out_batch_size = in_tile_area * out_channels * tile_count;
  const index_t out_stride = out_channels * tile_count;

  if (batch == 1) {
    Gemm(filter, input, in_tile_area, out_channels, in_channels, tile_count,
         output);
  } else {
#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < in_tile_area; ++i) {
        const float *in_ptr = input + b * in_batch_size + i * in_stride;
        const float *filter_ptr = filter + i * filter_stride;
        float *out_ptr = output + b * out_batch_size + i * out_stride;
        Gemm(filter_ptr, in_ptr, 1, out_channels, /* rows */
             in_channels,                         /* K */
             tile_count,                          /* cols */
             out_ptr);
      }
    }
  }
}

// NTOB => NToOB => NOHoWo
void TransformOutput4x4(const float *input,
                        index_t batch,
                        index_t out_height,
                        index_t out_width,
                        index_t out_channels,
                        index_t tile_count,
                        float *output) {
  const index_t stride = out_channels * tile_count;
  const index_t input_batch_size = 16 * stride;
  const index_t out_image_size = out_height * out_width;
  const index_t output_batch_size = out_channels * out_image_size;

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t m = 0; m < out_channels; ++m) {
      index_t tile_offset = 0;
      for (index_t h = 0; h < out_height; h += 2) {
        for (index_t w = 0; w < out_width; w += 2) {
          float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14,
              d15;
          float s0, s1, s2, s3, s4, s5, s6, s7;
          float v0, v1, v2, v3;

          const float *input_ptr =
              input + n * input_batch_size + m * tile_count + tile_offset;
          d0 = input_ptr[0];
          d1 = input_ptr[1 * stride];
          d2 = input_ptr[2 * stride];
          d3 = input_ptr[3 * stride];

          d4 = input_ptr[4 * stride];
          d5 = input_ptr[5 * stride];
          d6 = input_ptr[6 * stride];
          d7 = input_ptr[7 * stride];

          d8 = input_ptr[8 * stride];
          d9 = input_ptr[9 * stride];
          d10 = input_ptr[10 * stride];
          d11 = input_ptr[11 * stride];

          d12 = input_ptr[12 * stride];
          d13 = input_ptr[13 * stride];
          d14 = input_ptr[14 * stride];
          d15 = input_ptr[15 * stride];

          s0 = d0 + d1 + d2;
          s1 = d1 - d2 - d3;
          s2 = d4 + d5 + d6;
          s3 = d5 - d6 - d7;
          s4 = d8 + d9 + d10;
          s5 = d9 - d10 - d11;
          s6 = d12 + d13 + d14;
          s7 = d13 - d14 - d15;

          v0 = s0 + s2 + s4;
          v1 = s1 + s3 + s5;
          v2 = s2 - s4 - s6;
          v3 = s3 - s5 - s7;

          float *output_ptr = output + n * output_batch_size +
                              m * out_image_size + h * out_width + w;
          output_ptr[0] = v0;
          output_ptr[1] = v1;
          output_ptr[out_width] = v2;
          output_ptr[out_width + 1] = v3;

          ++tile_offset;
        }
      }
    }
  }
}

// NTOB => NToOB => NOHoWo
/**
 * AT =
⎡1  1  1   1    1   32  32   0⎤
⎢                             ⎥
⎢0  1  -1  2   -2   16  -16  0⎥
⎢                             ⎥
⎢0  1  1   4    4   8    8   0⎥
⎢                             ⎥
⎢0  1  -1  8   -8   4   -4   0⎥
⎢                             ⎥
⎢0  1  1   16  16   2    2   0⎥
⎢                             ⎥
⎣0  1  -1  32  -32  1   -1   1⎦
 *
 * @param input
 * @param batch
 * @param out_height
 * @param out_width
 * @param out_channels
 * @param tile_count
 * @param output
 */
void TransformOutput8x8(const float *input,
                        index_t batch,
                        index_t out_height,
                        index_t out_width,
                        index_t out_channels,
                        index_t tile_count,
                        float *output) {
  const index_t stride = out_channels * tile_count;
  const index_t input_batch_size = 64 * stride;
  const index_t out_image_size = out_height * out_width;
  const index_t output_batch_size = out_channels * out_image_size;

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t m = 0; m < out_channels; ++m) {
      index_t tile_offset = 0;
      float s[8][6];
      for (index_t h = 0; h < out_height; h += 6) {
        for (index_t w = 0; w < out_width; w += 6) {
          const float *input_ptr =
              input + n * input_batch_size + m * tile_count + tile_offset;
          for (int i = 0; i < 8; ++i) {
            float d0, d1, d2, d3, d4, d5, d6, d7;

            d0 = input_ptr[0];
            d1 = input_ptr[1 * stride];
            d2 = input_ptr[2 * stride];
            d3 = input_ptr[3 * stride];
            d4 = input_ptr[4 * stride];
            d5 = input_ptr[5 * stride];
            d6 = input_ptr[6 * stride];
            d7 = input_ptr[7 * stride];

            float u = d1 + d2;
            float v = d1 - d2;
            float w = d3 + d4;
            float x = d3 - d4;
            float y = d5 + d6;
            float z = d5 - d6;

            s[i][0] = d0 + u + w + y * 32;
            s[i][1] = v + x + x + z * 16;
            s[i][2] = u + w * 4 + y * 8;
            s[i][3] = v + x * 8 + z * 4;
            s[i][4] = u + w * 16 + y + y;
            s[i][5] = v + x * 32 + z + d7;

            input_ptr += 8 * stride;
          }

          float *output_ptr = output + n * output_batch_size +
                              m * out_image_size + h * out_width + w;

          for (int i = 0; i < 6; ++i) {
            float d0, d1, d2, d3, d4, d5, d6, d7;
            d0 = s[0][i];
            d1 = s[1][i];
            d2 = s[2][i];
            d3 = s[3][i];
            d4 = s[4][i];
            d5 = s[5][i];
            d6 = s[6][i];
            d7 = s[7][i];

            float u = d1 + d2;
            float v = d1 - d2;
            float w = d3 + d4;
            float x = d3 - d4;
            float y = d5 + d6;
            float z = d5 - d6;

            output_ptr[i] = d0 + u + w + y * 32;
            output_ptr[1 * out_width + i] = v + x + x + z * 16;
            output_ptr[2 * out_width + i] = u + w * 4 + y * 8;
            output_ptr[3 * out_width + i] = v + x * 8 + z * 4;
            output_ptr[4 * out_width + i] = u + w * 16 + y + y;
            output_ptr[5 * out_width + i] = v + x * 32 + z + d7;
          }

          ++tile_offset;
        }
      }
    }
  }
}
}  // namespace

// OCHW => TOC
// no need to optimize, it will exist in converter
void TransformFilter4x4(const float *filter,
                        const index_t in_channels,
                        const index_t out_channels,
                        float *output) {
  const index_t stride = out_channels * in_channels;

#pragma omp parallel for collapse(2)
  for (index_t m = 0; m < out_channels; ++m) {
    for (index_t c = 0; c < in_channels; ++c) {
      float g0, g1, g2, g3, g4, g5, g6, g7, g8;
      float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
          s15;

      // load filter
      index_t filter_offset = (m * in_channels + c) * 9;
      g0 = filter[filter_offset];
      g1 = filter[filter_offset + 1];
      g2 = filter[filter_offset + 2];
      g3 = filter[filter_offset + 3];
      g4 = filter[filter_offset + 4];
      g5 = filter[filter_offset + 5];
      g6 = filter[filter_offset + 6];
      g7 = filter[filter_offset + 7];
      g8 = filter[filter_offset + 8];

      // s = G * g * GT
      s0 = g0;
      s1 = (g0 + g2 + g1) * 0.5f;
      s2 = (g0 + g2 - g1) * 0.5f;
      s3 = g2;
      s4 = (g0 + g6 + g3) * 0.5f;
      s5 = ((g0 + g6 + g3) + (g2 + g8 + g5) + (g1 + g7 + g4)) * 0.25f;
      s6 = ((g0 + g6 + g3) + (g2 + g8 + g5) - (g1 + g7 + g4)) * 0.25f;
      s7 = (g2 + g8 + g5) * 0.5f;
      s8 = (g0 + g6 - g3) * 0.5f;
      s9 = ((g0 + g6 - g3) + (g2 + g8 - g5) + (g1 + g7 - g4)) * 0.25f;
      s10 = ((g0 + g6 - g3) + (g2 + g8 - g5) - (g1 + g7 - g4)) * 0.25f;
      s11 = (g2 + g8 - g5) * 0.5f;
      s12 = g6;
      s13 = (g6 + g8 + g7) * 0.5f;
      s14 = (g6 + g8 - g7) * 0.5f;
      s15 = g8;

      // store output
      index_t output_offset = m * in_channels + c;
      output[output_offset + 0 * stride] = s0;
      output[output_offset + 1 * stride] = s1;
      output[output_offset + 2 * stride] = s2;
      output[output_offset + 3 * stride] = s3;

      output[output_offset + 4 * stride] = s4;
      output[output_offset + 5 * stride] = s5;
      output[output_offset + 6 * stride] = s6;
      output[output_offset + 7 * stride] = s7;

      output[output_offset + 8 * stride] = s8;
      output[output_offset + 9 * stride] = s9;
      output[output_offset + 10 * stride] = s10;
      output[output_offset + 11 * stride] = s11;

      output[output_offset + 12 * stride] = s12;
      output[output_offset + 13 * stride] = s13;
      output[output_offset + 14 * stride] = s14;
      output[output_offset + 15 * stride] = s15;
    }
  }
}

// OCHW => TOC
// no need to optimize, it will exist in converter
/**
 * G =
⎡ 1      0      0  ⎤
⎢                  ⎥
⎢-2/9  -2/9   -2/9 ⎥
⎢                  ⎥
⎢-2/9   2/9   -2/9 ⎥
⎢                  ⎥
⎢1/90  1/45   2/45 ⎥
⎢                  ⎥
⎢1/90  -1/45  2/45 ⎥
⎢                  ⎥
⎢1/45  1/90   1/180⎥
⎢                  ⎥
⎢1/45  -1/90  1/180⎥
⎢                  ⎥
⎣ 0      0      1  ⎦
 *
 * @param filter
 * @param in_channels
 * @param out_channels
 * @param output
 */
void TransformFilter8x8(const float *filter,
                        const index_t in_channels,
                        const index_t out_channels,
                        float *output) {
  const index_t stride = out_channels * in_channels;

  const float G[8][3] = {{1.0f, 0.0f, 0.0f},
                         {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                         {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                         {1.0f / 90, 1.0f / 45, 2.0f / 45},
                         {1.0f / 90, -1.0f / 45, 2.0f / 45},
                         {1.0f / 45, 1.0f / 90, 1.0f / 180},
                         {1.0f / 45, -1.0f / 90, 1.0f / 180},
                         {0.0f, 0.0f, 1.0f}};

#pragma omp parallel for collapse(2)
  for (index_t m = 0; m < out_channels; ++m) {
    for (index_t c = 0; c < in_channels; ++c) {
      // load filter
      index_t filter_offset = (m * in_channels + c) * 9;
      float g0, g1, g2, g3, g4, g5, g6, g7, g8;
      g0 = filter[filter_offset];
      g1 = filter[filter_offset + 1];
      g2 = filter[filter_offset + 2];
      g3 = filter[filter_offset + 3];
      g4 = filter[filter_offset + 4];
      g5 = filter[filter_offset + 5];
      g6 = filter[filter_offset + 6];
      g7 = filter[filter_offset + 7];
      g8 = filter[filter_offset + 8];

      float s[3][8];
      for (int i = 0; i < 8; ++i) {
        s[0][i] = g0 * G[i][0] + g1 * G[i][1] + g2 * G[i][2];
        s[1][i] = g3 * G[i][0] + g4 * G[i][1] + g5 * G[i][2];
        s[2][i] = g6 * G[i][0] + g7 * G[i][1] + g8 * G[i][2];
      }

      // store output
      index_t output_offset = m * in_channels + c;
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          output[output_offset + (i * 8 + j) * stride] =
              G[i][0] * s[0][j] + G[i][1] * s[1][j] + G[i][2] * s[2][j];
        }
      }
    }
  }
}

void WinoGradConv3x3s1(const float *input,
                       const float *transformed_filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       const int out_tile_size,
                       float *transformed_input,
                       float *transformed_output,
                       float *output) {
  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;
  index_t tile_height_count =
      RoundUpDiv(out_height, static_cast<index_t>(out_tile_size));
  index_t tile_width_count =
      RoundUpDiv(out_width, static_cast<index_t>(out_tile_size));
  index_t tile_count = tile_height_count * tile_width_count;

  switch (out_tile_size) {
    case 2:
      TransformInput4x4(input, batch, in_height, in_width, in_channels,
                        tile_count, transformed_input);
      break;
    case 6:
      TransformInput8x8(input, batch, in_height, in_width, in_channels,
                        tile_count, transformed_input);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }

  BatchGemm(transformed_input, transformed_filter, batch, in_channels,
            out_channels, tile_count, out_tile_size, transformed_output);

  switch (out_tile_size) {
    case 2:
      TransformOutput4x4(transformed_output, batch, out_height, out_width,
                         out_channels, tile_count, output);
      break;
    case 6:
      TransformOutput8x8(transformed_output, batch, out_height, out_width,
                         out_channels, tile_count, output);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }
}

void WinoGradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       const int out_tile_size,
                       float *output) {
  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;
  index_t tile_height_count =
      RoundUpDiv(out_height, static_cast<index_t>(out_tile_size));
  index_t tile_width_count =
      RoundUpDiv(out_width, static_cast<index_t>(out_tile_size));
  index_t tile_count = tile_height_count * tile_width_count;
  index_t in_tile_area = (out_tile_size + 2) * (out_tile_size + 2);
  index_t transformed_input_size =
      in_tile_area * batch * in_channels * tile_count;
  index_t transformed_filter_size = in_tile_area * out_channels * in_channels;
  index_t transformed_output_size =
      in_tile_area * batch * out_channels * tile_count;

  float *transformed_input = new float[transformed_input_size];    // TNCB
  float *transformed_filter = new float[transformed_filter_size];  // TOC
  float *transformed_output = new float[transformed_output_size];

  switch (out_tile_size) {
    case 2:
      TransformFilter4x4(filter, in_channels, out_channels, transformed_filter);
      break;
    case 6:
      TransformFilter8x8(filter, in_channels, out_channels, transformed_filter);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }

  WinoGradConv3x3s1(input, transformed_filter, batch, in_height, in_width,
                    in_channels, out_channels, out_tile_size, transformed_input,
                    transformed_output, output);

  delete[] transformed_input;
  delete[] transformed_filter;
  delete[] transformed_output;
}

void ConvRef3x3s1(const float *input,
                  const float *filter,
                  const index_t batch,
                  const index_t in_height,
                  const index_t in_width,
                  const index_t in_channels,
                  const index_t out_channels,
                  float *output) {
  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;

#pragma omp parallel for collapse(4)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t m = 0; m < out_channels; ++m) {
      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          index_t out_offset =
              ((b * out_channels + m) * out_height + h) * out_width + w;
          output[out_offset] = 0;
          for (index_t c = 0; c < in_channels; ++c) {
            for (index_t kh = 0; kh < 3; ++kh) {
              for (index_t kw = 0; kw < 3; ++kw) {
                index_t ih = h + kh;
                index_t iw = w + kw;
                index_t in_offset =
                    ((b * in_channels + c) * in_height + ih) * in_width + iw;
                index_t filter_offset =
                    (((m * in_channels) + c) * 3 + kh) * 3 + kw;
                output[out_offset] += input[in_offset] * filter[filter_offset];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace mace

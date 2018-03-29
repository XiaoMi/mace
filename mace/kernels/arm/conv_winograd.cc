//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include <math.h>
#include <algorithm>

#include "mace/kernels/arm/conv_winograd.h"
#include "mace/kernels/gemm.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

namespace {
// NCHW => TNCB (T: in tile pixels, B: tile indices)
void TransformInput(const float *input,
                    const index_t batch,
                    const index_t in_height,
                    const index_t in_width,
                    const index_t in_channels,
                    const index_t tile_count,
                    float *output) {
  const index_t stride = batch * in_channels * tile_count;
  const index_t in_height_width = in_height * in_width;

#pragma omp parallel for
  for (index_t nc = 0; nc < batch * in_channels; ++nc) {
    index_t tile_index = nc * tile_count;
    for (index_t h = 0; h < in_height - 2; h += 2) {
      for (index_t w = 0; w < in_width - 2; w += 2) {
        float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14,
          d15;
        float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
          s15;

        // load tile data
        const index_t tile_offset = nc * in_height_width + h * in_width + w;
        d0 = input[tile_offset];
        d1 = input[tile_offset + 1];
        d2 = input[tile_offset + 2];
        d3 = input[tile_offset + 3];

        d4 = input[tile_offset + in_width];
        d5 = input[tile_offset + in_width + 1];
        d6 = input[tile_offset + in_width + 2];
        d7 = input[tile_offset + in_width + 3];

        d8 = input[tile_offset + 2 * in_width];
        d9 = input[tile_offset + 2 * in_width + 1];
        d10 = input[tile_offset + 2 * in_width + 2];
        d11 = input[tile_offset + 2 * in_width + 3];

        d12 = input[tile_offset + 3 * in_width];
        d13 = input[tile_offset + 3 * in_width + 1];
        d14 = input[tile_offset + 3 * in_width + 2];
        d15 = input[tile_offset + 3 * in_width + 3];

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
        output[tile_index + 0 * stride] = s0;
        output[tile_index + 1 * stride] = s1;
        output[tile_index + 2 * stride] = s2;
        output[tile_index + 3 * stride] = s3;

        output[tile_index + 4 * stride] = s4;
        output[tile_index + 5 * stride] = s5;
        output[tile_index + 6 * stride] = s6;
        output[tile_index + 7 * stride] = s7;

        output[tile_index + 8 * stride] = s8;
        output[tile_index + 9 * stride] = s9;
        output[tile_index + 10 * stride] = s10;
        output[tile_index + 11 * stride] = s11;

        output[tile_index + 12 * stride] = s12;
        output[tile_index + 13 * stride] = s13;
        output[tile_index + 14 * stride] = s14;
        output[tile_index + 15 * stride] = s15;

        ++tile_index;
      }
    }
  }
}

// OCHW => TOC
// no need to optimize, it will exist in converter
void TransformFilter(const float *filter,
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

// TOC * TNCB => TNOB
void BatchGemm(const float *input,
               const float *filter,
               index_t batch,
               index_t in_channels,
               index_t out_channels,
               index_t tile_count,
               float *output) {
  const index_t in_stride = batch * in_channels * tile_count;
  const index_t in_channels_tile_count = in_channels * tile_count;
  const index_t filter_stride = out_channels * in_channels;
  const index_t out_stride = batch * out_channels * tile_count;
  const index_t out_channels_tile_count = out_channels * tile_count;

  if (batch == 1) {
    Gemm(filter, input, 16, out_channels, in_channels, tile_count, output);
  } else {
    for (int i = 0; i < 16; ++i) {
      for (int b = 0; b < batch; ++b) {
        const float
          *in_ptr = input + i * in_stride + b * in_channels_tile_count;
        const float *filter_ptr = filter + i * filter_stride;
        float *out_ptr = output + i * out_stride + b * out_channels_tile_count;
        Gemm(filter_ptr,
             in_ptr,
             1,
             out_channels,  /* rows */
             in_channels,   /* K */
             tile_count,    /* cols */
             out_ptr);
      }
    }
  }
}

// TNOB => ToNOB => NOHoWo
void TransformOutput(const float *input,
                     index_t batch,
                     index_t out_height,
                     index_t out_width,
                     index_t out_channels,
                     index_t tile_count,
                     float *output) {
  const index_t in_stride = batch * out_channels * tile_count;

#pragma omp parallel for
  for (index_t nm = 0; nm < batch * out_channels; ++nm) {
    index_t tile_offset = nm * tile_count;
    for (index_t h = 0; h < out_height; h += 2) {
      for (index_t w = 0; w < out_width; w += 2) {
        float d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14,
          d15;
        float s0, s1, s2, s3, s4, s5, s6, s7;
        float v0, v1, v2, v3;

        d0 = input[tile_offset + 0 * in_stride];
        d1 = input[tile_offset + 1 * in_stride];
        d2 = input[tile_offset + 2 * in_stride];
        d3 = input[tile_offset + 3 * in_stride];

        d4 = input[tile_offset + 4 * in_stride];
        d5 = input[tile_offset + 5 * in_stride];
        d6 = input[tile_offset + 6 * in_stride];
        d7 = input[tile_offset + 7 * in_stride];

        d8 = input[tile_offset + 8 * in_stride];
        d9 = input[tile_offset + 9 * in_stride];
        d10 = input[tile_offset + 10 * in_stride];
        d11 = input[tile_offset + 11 * in_stride];

        d12 = input[tile_offset + 12 * in_stride];
        d13 = input[tile_offset + 13 * in_stride];
        d14 = input[tile_offset + 14 * in_stride];
        d15 = input[tile_offset + 15 * in_stride];

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

        index_t out_offset = nm * out_height * out_width + h * out_width + w;
        output[out_offset] = v0;
        output[out_offset + 1] = v1;
        output[out_offset + out_width] = v2;
        output[out_offset + out_width + 1] = v3;

        ++tile_offset;
      }
    }
  }
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
                index_t
                  filter_offset = (((m * in_channels) + c) * 3 + kh) * 3 + kw;
                output[out_offset] += input[in_offset] * filter[filter_offset];
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace

void WinoGradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       float *transformed_input,
                       float *transformed_filter,
                       float *transformed_output,
                       bool is_filter_transformed,
                       float *output) {
  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;
  index_t tile_height_count = (out_height + 1) / 2;
  index_t tile_width_count = (out_width + 1) / 2;
  index_t tile_count = tile_height_count * tile_width_count;

  TransformInput(input,
                 batch,
                 in_height,
                 in_width,
                 in_channels,
                 tile_count,
                 transformed_input);

  // TODO(liyin): put it in model converter, but do not worry, it is fast and
  // will only do once
  if (!is_filter_transformed) {
    TransformFilter(filter, in_channels, out_channels, transformed_filter);
  }

  BatchGemm(transformed_input,
            transformed_filter,
            batch,
            in_channels,
            out_channels,
            tile_count,
            transformed_output);

  TransformOutput(transformed_output,
                  batch,
                  out_height,
                  out_width,
                  out_channels,
                  tile_count,
                  output);
}

void WinoGradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       float *output) {
  index_t out_height = in_height - 2;
  index_t out_width = in_width - 2;
  index_t tile_height_count = (out_height + 1) / 2;
  index_t tile_width_count = (out_width + 1) / 2;
  index_t tile_count = tile_height_count * tile_width_count;

  index_t transformed_input_size = 16 * batch * in_channels * tile_count;
  index_t transformed_filter_size = 16 * out_channels * in_channels;
  index_t transformed_output_size = 16 * batch * out_channels * tile_count;

  float *transformed_input = new float[transformed_input_size];  // TNCB
  float *transformed_filter = new float[transformed_filter_size];  // TOC
  float *transformed_output = new float[transformed_output_size];

  WinoGradConv3x3s1(input,
                    filter,
                    batch,
                    in_height,
                    in_width,
                    in_channels,
                    out_channels,
                    transformed_input,
                    transformed_filter,
                    transformed_output,
                    false,
                    output);

  delete[]transformed_input;
  delete[]transformed_filter;
  delete[]transformed_output;
}

}  // namespace kernels
}  // namespace mace

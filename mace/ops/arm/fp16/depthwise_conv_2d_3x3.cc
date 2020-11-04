// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include <arm_neon.h>

#include "mace/ops/arm/base/depthwise_conv_2d_3x3.h"

namespace mace {
namespace ops {
namespace arm {

template<typename float16_t>
void DepthwiseConv2d3x3Pixel(const float16_t *in_base,
                             const float16_t *filter,
                             const index_t out_h,
                             const index_t out_w,
                             const index_t in_h_start,
                             const index_t in_w_start,
                             const index_t out_width,
                             const index_t in_height,
                             const index_t in_width,
                             float16_t *out_base) {
  const index_t filter_width = 3;
  float sum = 0.0f;

  index_t in_h = in_h_start;
  const float16_t *in = in_base + in_h * in_width;
  const float16_t *filter_ptr = filter;
  if (in_h >= 0 && in_h < in_height) {
    index_t in_w = in_w_start;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[0];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[1];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[2];
    }
  }
  in_h++;
  in += in_width;
  filter_ptr += filter_width;
  if (in_h >= 0 && in_h < in_height) {
    index_t in_w = in_w_start;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[0];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[1];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[2];
    }
  }
  in_h++;
  in += in_width;
  filter_ptr += filter_width;
  if (in_h >= 0 && in_h < in_height) {
    index_t in_w = in_w_start;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[0];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[1];
    }
    in_w++;
    if (in_w >= 0 && in_w < in_width) {
      sum += in[in_w] * filter_ptr[2];
    }
  }
  out_base[out_h * out_width + out_w] = static_cast<float16_t>(sum);
}

template<>
MaceStatus DepthwiseConv2dK3x3S1<float16_t>::DoCompute(
    const DepthwiseConvComputeParam &p, const float16_t *filter_data,
    const float16_t *input_data, float16_t *output_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        const index_t c = m / p.multiplier;
        const index_t multi_index = m % p.multiplier;
        auto filter_ptr = filter_data + multi_index * p.in_channels * 9 + c * 9;
        auto in_base = input_data + b * p.in_batch_size + c * p.in_image_size;
        auto out_base = output_data + b * p.out_batch_size +
            m * p.out_image_size;
        index_t h, w;

        // top
        for (h = 0; h < p.valid_h_start; ++h) {
          for (w = 0; w < p.out_width; ++w) {
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h,
                                    w,
                                    h - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
          }
        }

        // load filter (1 outch x 3 height x 3 width): vf_outch_height
        float16x8_t vf00, vf01;
        vf00 = vld1q_f16(filter_ptr);
        vf01 = vld1q_f16(filter_ptr + 8);

        for (h = p.valid_h_start; h + 1 < p.valid_h_stop; h += 2) {
          // left
          for (w = 0; w < p.valid_w_start; ++w) {
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h,
                                    w,
                                    h - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h + 1,
                                    w,
                                    h + 1 - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
          }

          for (w = p.valid_w_start; w + 7 < p.valid_w_stop; w += 8) {
            // input (4 height x 3 slide): vi_height_slide
            float16x8_t vi00, vi01, vi02, vi0n;
            float16x8_t vi10, vi11, vi12, vi1n;
            float16x8_t vi20, vi21, vi22, vi2n;
            float16x8_t vi30, vi31, vi32, vi3n;

            // output (1 outch x 2 height x 8 width): vo_outch_height
            float16x8_t vo00, vo01;

            // load input
            index_t in_h = h - p.pad_top;
            index_t in_w = w - p.pad_left;
            index_t in_offset = in_h * p.in_width + in_w;
            vi00 = vld1q_f16(in_base + in_offset);
            vi0n = vld1q_f16(in_base + in_offset + 8);
            vi10 = vld1q_f16(in_base + in_offset + p.in_width);
            vi1n = vld1q_f16(in_base + in_offset + p.in_width + 8);
            vi20 = vld1q_f16(in_base + in_offset + 2 * p.in_width);
            vi2n = vld1q_f16(in_base + in_offset + 2 * p.in_width + 8);
            vi30 = vld1q_f16(in_base + in_offset + 3 * p.in_width);
            vi3n = vld1q_f16(in_base + in_offset + 3 * p.in_width + 8);

            vi01 = vextq_f16(vi00, vi0n, 1);
            vi02 = vextq_f16(vi00, vi0n, 2);
            vi11 = vextq_f16(vi10, vi1n, 1);
            vi12 = vextq_f16(vi10, vi1n, 2);
            vi21 = vextq_f16(vi20, vi2n, 1);
            vi22 = vextq_f16(vi20, vi2n, 2);
            vi31 = vextq_f16(vi30, vi3n, 1);
            vi32 = vextq_f16(vi30, vi3n, 2);

            // load ouptut
            index_t out_offset = h * p.out_width + w;
            vo00 = vld1q_f16(out_base + out_offset);
            vo01 = vld1q_f16(out_base + out_offset + p.out_width);

            // outch 0, height 0
            vo00 = vfmaq_laneq_f16(vo00, vi00, vf00, 0);
            vo00 = vfmaq_laneq_f16(vo00, vi01, vf00, 1);
            vo00 = vfmaq_laneq_f16(vo00, vi02, vf00, 2);
            vo00 = vfmaq_laneq_f16(vo00, vi10, vf00, 3);
            vo00 = vfmaq_laneq_f16(vo00, vi11, vf00, 4);
            vo00 = vfmaq_laneq_f16(vo00, vi12, vf00, 5);
            vo00 = vfmaq_laneq_f16(vo00, vi20, vf00, 6);
            vo00 = vfmaq_laneq_f16(vo00, vi21, vf00, 7);
            vo00 = vfmaq_laneq_f16(vo00, vi22, vf01, 0);

            // outch 0, height 1
            vo01 = vfmaq_laneq_f16(vo01, vi10, vf00, 0);
            vo01 = vfmaq_laneq_f16(vo01, vi11, vf00, 1);
            vo01 = vfmaq_laneq_f16(vo01, vi12, vf00, 2);
            vo01 = vfmaq_laneq_f16(vo01, vi20, vf00, 3);
            vo01 = vfmaq_laneq_f16(vo01, vi21, vf00, 4);
            vo01 = vfmaq_laneq_f16(vo01, vi22, vf00, 5);
            vo01 = vfmaq_laneq_f16(vo01, vi30, vf00, 6);
            vo01 = vfmaq_laneq_f16(vo01, vi31, vf00, 7);
            vo01 = vfmaq_laneq_f16(vo01, vi32, vf01, 0);
            vst1q_f16(out_base + out_offset, vo00);
            vst1q_f16(out_base + out_offset + p.out_width, vo01);
          }  // w

          // right
          for (; w < p.out_width; ++w) {
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h,
                                    w,
                                    h - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h + 1,
                                    w,
                                    h + 1 - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
          }
        }  // h

        // bottom
        for (; h < p.out_height; ++h) {
          for (w = 0; w < p.out_width; ++w) {
            DepthwiseConv2d3x3Pixel(in_base,
                                    filter_ptr,
                                    h,
                                    w,
                                    h - p.pad_top,
                                    w - p.pad_left,
                                    p.out_width,
                                    p.in_height,
                                    p.in_width,
                                    out_base);
          }
        }
      }  // m
    }    // b
  }, 0, p.batch, 1, 0, p.out_channels, 1);  // threadpool

  return MaceStatus::MACE_SUCCESS;
}

template<>
MaceStatus DepthwiseConv2dK3x3S2<float16_t>::DoCompute(
    const DepthwiseConvComputeParam &p, const float16_t *filter_data,
    const float16_t *input_data, float16_t *output_data) {
  p.thread_pool.Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t m = start1; m < end1; m += step1) {
            index_t c = m / p.multiplier;
            index_t multi_index = m % p.multiplier;
            auto filter_ptr = filter_data + multi_index * p.in_channels * 9 +
                c * 9;
            auto in_base = input_data + b * p.in_batch_size +
                c * p.in_image_size;
            auto out_base = output_data + b * p.out_batch_size +
                m * p.out_image_size;
            index_t h, w;

            // top
            for (h = 0; h < p.valid_h_start; ++h) {
              for (w = 0; w < p.out_width; ++w) {
                DepthwiseConv2d3x3Pixel(in_base,
                                        filter_ptr,
                                        h,
                                        w,
                                        h * 2 - p.pad_top,
                                        w * 2 - p.pad_left,
                                        p.out_width,
                                        p.in_height,
                                        p.in_width,
                                        out_base);
              }
            }

            // load filter (1 outch x 3 height x 3 width): vf_outch_height
            float16x8_t vf00, vf01;
            vf00 = vld1q_f16(filter_ptr);
            vf01 = vld1q_f16(filter_ptr + 8);

            for (h = p.valid_h_start; h < p.valid_h_stop; ++h) {
              // left
              for (w = 0; w < p.valid_w_start; ++w) {
                DepthwiseConv2d3x3Pixel(in_base,
                                        filter_ptr,
                                        h,
                                        w,
                                        h * 2 - p.pad_top,
                                        w * 2 - p.pad_left,
                                        p.out_width,
                                        p.in_height,
                                        p.in_width,
                                        out_base);
              }

              for (w = p.valid_w_start; w + 3 < p.valid_w_stop; w += 8) {
                float16x8x2_t vi0, vi1, vi2;
                float16x8_t vi0n, vi1n, vi2n;

                // input (3 height x 3 slide): vi_height_slide
                float16x8_t vi00, vi01, vi02;
                float16x8_t vi10, vi11, vi12;
                float16x8_t vi20, vi21, vi22;

                // output (1 outch x 1 height x 8 width): vo
                float16x8_t vo;

                // load input
                index_t in_h = h * 2 - p.pad_top;
                index_t in_w = w * 2 - p.pad_left;
                index_t in_offset = in_h * p.in_width + in_w;
                vi0 = vld2q_f16(in_base + in_offset);  // [0.2.4.6, 1.3.5.7]
                vi1 = vld2q_f16(in_base + in_offset + p.in_width);
                vi2 = vld2q_f16(in_base + in_offset + 2 * p.in_width);

                vi0n = vld1q_f16(in_base + in_offset + 16);  // [8.9.10.11]
                vi1n = vld1q_f16(in_base + in_offset + p.in_width + 16);
                vi2n = vld1q_f16(in_base + in_offset + 2 * p.in_width + 16);

                // load ouptut
                index_t out_offset = h * p.out_width + w;
                vo = vld1q_f16(out_base + out_offset);

                vi00 = vi0.val[0];                // [0.2.4.6]
                vi01 = vi0.val[1];                // [1.3.5.7]
                vi02 = vextq_f16(vi00, vi0n, 1);  // [2.4.6.8]
                vi10 = vi1.val[0];
                vi11 = vi1.val[1];
                vi12 = vextq_f16(vi10, vi1n, 1);
                vi20 = vi2.val[0];
                vi21 = vi2.val[1];
                vi22 = vextq_f16(vi20, vi2n, 1);

                // outch 0, height 0
                vo = vfmaq_laneq_f16(vo, vi00, vf00, 0);
                vo = vfmaq_laneq_f16(vo, vi01, vf00, 1);
                vo = vfmaq_laneq_f16(vo, vi02, vf00, 2);
                vo = vfmaq_laneq_f16(vo, vi10, vf00, 3);
                vo = vfmaq_laneq_f16(vo, vi11, vf00, 4);
                vo = vfmaq_laneq_f16(vo, vi12, vf00, 5);
                vo = vfmaq_laneq_f16(vo, vi20, vf00, 6);
                vo = vfmaq_laneq_f16(vo, vi21, vf00, 7);
                vo = vfmaq_laneq_f16(vo, vi22, vf01, 0);
                vst1q_f16(out_base + out_offset, vo);
              }  // w

              // right
              for (; w < p.out_width; ++w) {
                DepthwiseConv2d3x3Pixel(in_base,
                                        filter_ptr,
                                        h,
                                        w,
                                        h * 2 - p.pad_top,
                                        w * 2 - p.pad_left,
                                        p.out_width,
                                        p.in_height,
                                        p.in_width,
                                        out_base);
              }
            }  // h

            // bottom
            for (; h < p.out_height; ++h) {
              for (w = 0; w < p.out_width; ++w) {
                DepthwiseConv2d3x3Pixel(in_base,
                                        filter_ptr,
                                        h,
                                        w,
                                        h * 2 - p.pad_top,
                                        w * 2 - p.pad_left,
                                        p.out_width,
                                        p.in_height,
                                        p.in_width,
                                        out_base);
              }
            }
          }  // m
        }    // b
      },
      0, p.batch, 1, 0, p.out_channels, 1);

  return MaceStatus::MACE_SUCCESS;
}

void RegisterFP16DepthwiseConv2dK3x3Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_FP16_DELEGATOR(
      registry, DepthwiseConv2dK3x3S1<float16_t>,
      delegator::DepthwiseConv2dParam,
      MACE_DELEGATOR_KEY_EX(DepthwiseConv2d, DeviceType::CPU,
                            float16_t, ImplType::NEON, K3x3S1));
  MACE_REGISTER_FP16_DELEGATOR(
      registry, DepthwiseConv2dK3x3S2<float16_t>,
      delegator::DepthwiseConv2dParam,
      MACE_DELEGATOR_KEY_EX(DepthwiseConv2d, DeviceType::CPU,
                            float16_t, ImplType::NEON, K3x3S2));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace


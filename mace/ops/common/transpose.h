// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_COMMON_TRANSPOSE_H_
#define MACE_OPS_COMMON_TRANSPOSE_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif  // MACE_ENABLE_NEON
#include <algorithm>
#include <vector>
#include "mace/core/op_context.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {

template<typename T>
void TransposeNHWCToNCHWC3(utils::ThreadPool *thread_pool,
                           const T *input,
                           T *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;

  thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t h = start; h < end; h += step) {
      index_t in_offset = h * width * 3;
      index_t out_offset = h * width;

      for (index_t w = 0; w < width; ++w) {
        for (index_t c = 0; c < 3; ++c) {
          output[out_offset + c * image_size + w] =
              input[in_offset + w * 3 + c];
        }
      }
    }
  }, 0, height, 1);
}

template<>
inline void TransposeNHWCToNCHWC3<float>(utils::ThreadPool *thread_pool,
                                         const float *input,
                                         float *output,
                                         const index_t height,
                                         const index_t width) {
  index_t image_size = height * width;

  thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t h = start; h < end; h += step) {
      index_t in_offset = h * width * 3;
      index_t out_offset = h * width;

#if defined(MACE_ENABLE_NEON)
      index_t w;
      for (w = 0; w + 3 < width; w += 4) {
        float32x4x3_t vi = vld3q_f32(input + in_offset);
        vst1q_f32(output + out_offset, vi.val[0]);
        vst1q_f32(output + out_offset + image_size, vi.val[1]);
        vst1q_f32(output + out_offset + image_size * 2, vi.val[2]);

        in_offset += 12;
        out_offset += 4;
      }
      for (; w < width; ++w) {
        for (index_t c = 0; c < 3; ++c) {
          output[h * width + image_size * c + w] =
              input[h * width * 3 + w * 3 + c];
        }
      }
#else
      for (index_t w = 0; w < width; ++w) {
        for (index_t c = 0; c < 3; ++c) {
          output[out_offset + c * image_size + w] =
              input[in_offset + w * 3 + c];
        }
      }
#endif
    }
  }, 0, height, 1);
}

template<typename T>
void TransposeNCHWToNHWCC2(utils::ThreadPool *thread_pool,
                           const T *input,
                           T *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;

  thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t h = start; h < end; h += step) {
      index_t in_offset = h * width;
      index_t out_offset = h * width * 2;

      for (index_t w = 0; w < width; ++w) {
        for (index_t c = 0; c < 2; ++c) {
          output[out_offset + w * 2 + c] =
              input[in_offset + c * image_size + w];
        }
      }
    }
  }, 0, height, 1);
}

template<>
inline void TransposeNCHWToNHWCC2<float>(utils::ThreadPool *thread_pool,
                                         const float *input,
                                         float *output,
                                         const index_t height,
                                         const index_t width) {
  index_t image_size = height * width;

  thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t h = start; h < end; h += step) {
      index_t in_offset = h * width;
      index_t out_offset = h * width * 2;

#if defined(MACE_ENABLE_NEON)
      index_t w;
      for (w = 0; w + 3 < width; w += 4) {
        float32x4_t vi0 = vld1q_f32(input + in_offset);
        float32x4_t vi1 = vld1q_f32(input + in_offset + image_size);
        float32x4x2_t vi = {{vi0, vi1}};
        vst2q_f32(output + out_offset, vi);
        in_offset += 4;
        out_offset += 8;
      }
      for (; w < width; ++w) {
        for (index_t c = 0; c < 2; ++c) {
          output[h * width * 2 + w * 2 + c] =
              input[h * width + image_size * c + w];
        }
      }
#else
      for (index_t w = 0; w < width; ++w) {
        for (index_t c = 0; c < 2; ++c) {
          output[out_offset + w * 2 + c] =
              input[in_offset + c * image_size + w];
        }
      }
#endif
    }
  }, 0, height, 1);
}

template<typename T>
MaceStatus Transpose(utils::ThreadPool *thread_pool,
                     const T *input,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int> &dst_dims,
                     T *output) {
  MACE_CHECK((input_shape.size() == 2 && dst_dims.size() == 2) ||
      (input_shape.size() == 4 && dst_dims.size() == 4),
             "Only support 2D or 4D transpose");

  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }

  if (input_shape.size() == 2) {
    MACE_CHECK(dst_dims[0] == 1 && dst_dims[1] == 0, "no need transform");
    index_t height = input_shape[0];
    index_t width = input_shape[1];
    index_t stride_i = height;
    index_t stride_j = width;
    index_t tile_size = height > 512 || width > 512 ? 64 : 32;

    thread_pool->Compute2D([=](index_t start0, index_t end0, index_t step0,
                               index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        for (index_t j = start1; j < end1; j += step1) {
          index_t end_i = std::min(i + tile_size, height);
          index_t end_j = std::min(j + tile_size, width);
          for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
            for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
              output[tile_j * stride_i + tile_i] =
                  input[tile_i * stride_j + tile_j];
            }
          }
        }
      }
    }, 0, height, tile_size, 0, width, tile_size);
  } else if (input_shape.size() == 4) {
    std::vector<int> transpose_order_from_NHWC_to_NCHW{0, 3, 1, 2};
    std::vector<int> transpose_order_from_NCHW_to_NHWC{0, 2, 3, 1};
    index_t batch_size = input_shape[1] * input_shape[2] * input_shape[3];

    if (dst_dims == transpose_order_from_NHWC_to_NCHW && input_shape[3] == 3) {
      for (index_t b = 0; b < input_shape[0]; ++b) {
        TransposeNHWCToNCHWC3(thread_pool,
                              input + b * batch_size,
                              output + b * batch_size,
                              input_shape[1],
                              input_shape[2]);
      }
    } else if (dst_dims == transpose_order_from_NCHW_to_NHWC
        && input_shape[1] == 2) {
      for (index_t b = 0; b < input_shape[0]; ++b) {
        TransposeNCHWToNHWCC2(thread_pool,
                              input + b * batch_size,
                              output + b * batch_size,
                              input_shape[2],
                              input_shape[3]);
      }
    } else if (dst_dims == std::vector<int>{0, 2, 1, 3}) {
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t channel = input_shape[3];
      index_t channel_raw_size = channel * sizeof(T);
      index_t stride_i = height;
      index_t stride_j = width;
      index_t tile_size = std::max(static_cast<index_t>(1),
                                   static_cast<index_t>(std::sqrt(
                                       8 * 1024 / channel)));
      for (index_t i = 0; i < height; i += tile_size) {
        for (index_t j = 0; j < width; j += tile_size) {
          index_t end_i = std::min(i + tile_size, height);
          index_t end_j = std::min(j + tile_size, width);
          for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
            for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
              memcpy(output + (tile_j * stride_i + tile_i) * channel,
                     input + (tile_i * stride_j + tile_j) * channel,
                     channel_raw_size);
            }
          }
        }
      }
    } else {
      std::vector<index_t>
          in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                    input_shape[2] * input_shape[3], input_shape[3], 1};
      std::vector<index_t>
          out_stride{output_shape[1] * output_shape[2] * output_shape[3],
                     output_shape[2] * output_shape[3], output_shape[3], 1};

      std::vector<index_t> idim(4, 0);
      std::vector<index_t> odim(4, 0);
      for (odim[0] = 0; odim[0] < output_shape[0]; ++odim[0]) {
        for (odim[1] = 0; odim[1] < output_shape[1]; ++odim[1]) {
          for (odim[2] = 0; odim[2] < output_shape[2]; ++odim[2]) {
            for (odim[3] = 0; odim[3] < output_shape[3]; ++odim[3]) {
              idim[dst_dims[0]] = odim[0];
              idim[dst_dims[1]] = odim[1];
              idim[dst_dims[2]] = odim[2];
              idim[dst_dims[3]] = odim[3];

              output[odim[0] * out_stride[0] + odim[1] * out_stride[1]
                  + odim[2] * out_stride[2] + odim[3]] =
                  input[idim[0] * in_stride[0] + idim[1] * in_stride[1]
                      + idim[2] * in_stride[2] + idim[3]];
            }
          }
        }
      }
    }
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_TRANSPOSE_H_

// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_FP32_CONV_2D_3X3_WINOGRAD_H_
#define MACE_OPS_ARM_FP32_CONV_2D_3X3_WINOGRAD_H_

#include <memory>

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/arm/fp32/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Conv2dK3x3Winograd : public Conv2dBase {
 public:
  Conv2dK3x3Winograd(int pad_top, int pad_bottom, int pad_left, int pad_right)
      : gemm_(),
        pad_top_(pad_top),
        pad_bottom_(pad_bottom),
        pad_left_(pad_left),
        pad_right_(pad_right),
        transformed_filter_(nullptr),
        out_tile_size_(0) {}

  virtual ~Conv2dK3x3Winograd() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output);

 private:
  void UnPackOutput(const Tensor &padded_output,
                    Tensor *output);

  void TransformFilter4x4(const float *filter,
                          const index_t in_channels,
                          const index_t out_channels,
                          float *output);

  void TransformFilter8x8(const float *filter,
                          const index_t in_channels,
                          const index_t out_channels,
                          float *output);

  void TransformInput4x4(const float *input,
                         const index_t batch,
                         const index_t in_height,
                         const index_t in_width,
                         const index_t in_channels,
                         const index_t tile_count,
                         float *output);

  void TransformInput8x8(const float *input,
                         const index_t batch,
                         const index_t in_height,
                         const index_t in_width,
                         const index_t in_channels,
                         const index_t tile_count,
                         float *output);

  void TransformOutput4x4(const float *input,
                          index_t batch,
                          index_t out_height,
                          index_t out_width,
                          index_t out_channels,
                          index_t tile_count,
                          float *output);

  void TransformOutput8x8(const float *input,
                          index_t batch,
                          index_t out_height,
                          index_t out_width,
                          index_t out_channels,
                          index_t tile_count,
                          float *output);

  Gemm gemm_;
  int pad_top_;
  int pad_bottom_;
  int pad_left_;
  int pad_right_;
  std::unique_ptr<Tensor> transformed_filter_;
  index_t out_tile_size_;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_CONV_2D_3X3_WINOGRAD_H_

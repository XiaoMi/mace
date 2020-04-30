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

#ifndef MACE_OPS_ARM_BASE_CONV_2D_H_
#define MACE_OPS_ARM_BASE_CONV_2D_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/gemm.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/conv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

struct ConvComputeParam {
  const index_t batch;
  const index_t in_channels;
  const index_t in_height;
  const index_t in_width;
  const index_t out_channels;
  const index_t out_height;
  const index_t out_width;

  const index_t in_image_size;
  const index_t out_image_size;
  const index_t in_batch_size;
  const index_t out_batch_size;

  utils::ThreadPool &thread_pool;

  ConvComputeParam(const index_t b,
                   const index_t in_c,
                   const index_t in_h,
                   const index_t in_w,
                   const index_t out_c,
                   const index_t out_h,
                   const index_t out_w,
                   const index_t in_size,
                   const index_t out_size,
                   const index_t in_b_size,
                   const index_t out_b_size,
                   utils::ThreadPool *thrd_pool)
      : batch(b), in_channels(in_c), in_height(in_h), in_width(in_w),
        out_channels(out_c), out_height(out_h), out_width(out_w),
        in_image_size(in_size), out_image_size(out_size),
        in_batch_size(in_b_size), out_batch_size(out_b_size),
        thread_pool(*thrd_pool) {}
};

struct DepthwiseConvComputeParam : public ConvComputeParam {
  const int pad_top;
  const int pad_left;
  const index_t multiplier;
  const index_t valid_h_start;
  const index_t valid_h_stop;
  const index_t valid_w_start;
  const index_t valid_w_stop;
  DepthwiseConvComputeParam(const index_t b,
                            const index_t in_c,
                            const index_t in_h,
                            const index_t in_w,
                            const index_t out_c,
                            const index_t out_h,
                            const index_t out_w,
                            const index_t in_size,
                            const index_t out_size,
                            const index_t in_b_size,
                            const index_t out_b_size,
                            utils::ThreadPool *thrd_pool,
                            const int pad_top_data,
                            const int pad_left_data,
                            const index_t multiplier_data,
                            const index_t valid_height_start,
                            const index_t valid_height_stop,
                            const index_t valid_width_start,
                            const index_t valid_width_stop)
      : ConvComputeParam(b, in_c, in_h, in_w, out_c, out_h, out_w,
                         in_size, out_size, in_b_size, out_b_size, thrd_pool),
        pad_top(pad_top_data), pad_left(pad_left_data),
        multiplier(multiplier_data),
        valid_h_start(valid_height_start), valid_h_stop(valid_height_stop),
        valid_w_start(valid_width_start), valid_w_stop(valid_width_stop) {}
};

class Conv2dBase : public delegator::Conv2d {
 public:
  explicit Conv2dBase(const delegator::Conv2dParam &param, int type_size)
      : delegator::Conv2d(param), type_size_(type_size) {}

  virtual ~Conv2dBase() = default;

 protected:
  void CalOutputShapeAndInputPadSize(const std::vector<index_t> &input_shape,
                                     const std::vector<index_t> &filter_shape,
                                     std::vector<index_t> *output_shape,
                                     std::vector<int> *in_pad_size);

  void CalOutputBoundaryWithoutUsingInputPad(const std::vector<index_t>
                                             &output_shape,
                                             const std::vector<int>
                                             in_pad_size,
                                             std::vector<index_t>
                                             *out_bound);

  void CalOutputShapeAndPadSize(const Tensor *input,
                                const Tensor *filter,
                                const int out_tile_height,
                                const int out_tile_width,
                                std::vector<index_t> *output_shape,
                                std::vector<int> *in_pad_size,
                                std::vector<int> *out_pad_size);

  MaceStatus ResizeOutAndPadInOut(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output,
                                  const int out_tile_height,
                                  const int out_tile_width,
                                  std::unique_ptr<const Tensor> *padded_input,
                                  std::unique_ptr<Tensor> *padded_output);

  void PadInput(const Tensor &src,
                const int pad_top,
                const int pad_left,
                Tensor *dst);
  void UnPadOutput(const Tensor &src, Tensor *dst);

  ConvComputeParam PreWorkAndGetConv2DParam(
      const OpContext *context, const Tensor *in_tensor, Tensor *out_tensor);
  DepthwiseConvComputeParam PreWorkAndGetDepthwiseConv2DParam(
      const OpContext *context, const Tensor *input,
      const Tensor *filter, Tensor *output);

 private:
  int type_size_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_CONV_2D_H_

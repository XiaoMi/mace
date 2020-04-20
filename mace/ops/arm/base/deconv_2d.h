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

#ifndef MACE_OPS_ARM_BASE_DECONV_2D_H_
#define MACE_OPS_ARM_BASE_DECONV_2D_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/ops/arm/base/gemm.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/deconv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

struct DeconvComputeParam {
  const index_t batch;
  const index_t in_channels;
  const index_t in_height;
  const index_t in_width;
  const index_t out_channels;
  const index_t out_height;
  const index_t out_width;
  const index_t out_img_size;

  utils::ThreadPool &thread_pool;

  DeconvComputeParam(const index_t b,
                     const index_t in_c,
                     const index_t in_h,
                     const index_t in_w,
                     const index_t out_c,
                     const index_t out_h,
                     const index_t out_w,
                     const index_t out_size,
                     utils::ThreadPool *thrd_pool)
      : batch(b), in_channels(in_c), in_height(in_h), in_width(in_w),
        out_channels(out_c), out_height(out_h), out_width(out_w),
        out_img_size(out_size), thread_pool(*thrd_pool) {}
};

struct DepthwiseDeconvComputeParam {
  const index_t batch;
  const index_t in_channels;
  const index_t in_height;
  const index_t in_width;
  const index_t in_img_size;
  const index_t out_height;
  const index_t out_width;
  const index_t out_img_size;
  utils::ThreadPool &thread_pool;

  DepthwiseDeconvComputeParam(const index_t b,
                              const index_t in_c,
                              const index_t in_h,
                              const index_t in_w,
                              const index_t in_size,
                              const index_t out_h,
                              const index_t out_w,
                              const index_t out_size,
                              utils::ThreadPool *thrd_pool)
      : batch(b),
        in_channels(in_c),
        in_height(in_h),
        in_width(in_w),
        in_img_size(in_size),
        out_height(out_h),
        out_width(out_w),
        out_img_size(out_size),
        thread_pool(*thrd_pool) {}
};

struct GroupDeconvComputeParam {
  const index_t batch;
  const index_t in_channels;
  const index_t in_height;
  const index_t in_width;

  const index_t out_channels;
  const index_t out_height;
  const index_t out_width;

  const index_t in_img_size;
  const index_t out_img_size;

  const index_t inch_g;
  const index_t outch_g;
  utils::ThreadPool &thread_pool;

  GroupDeconvComputeParam(const index_t in_b,
                          const index_t in_ch,
                          const index_t in_h,
                          const index_t in_w,
                          const index_t out_ch,
                          const index_t out_h,
                          const index_t out_w,
                          const index_t in_size,
                          const index_t out_size,
                          const index_t in_ch_g,
                          const index_t out_ch_g,
                          utils::ThreadPool *thrd_pool)
      : batch(in_b),
        in_channels(in_ch),
        in_height(in_h),
        in_width(in_w),
        out_channels(out_ch),
        out_height(out_h),
        out_width(out_w),
        in_img_size(in_size),
        out_img_size(out_size),
        inch_g(in_ch_g),
        outch_g(out_ch_g),
        thread_pool(*thrd_pool) {}
};

class Deconv2dBase : public delegator::Deconv2d {
 public:
  explicit Deconv2dBase(const delegator::Deconv2dParam &param, int type_size)
      : delegator::Deconv2d(param),
        group_(param.group_), type_size_(type_size) {}

  virtual ~Deconv2dBase() = default;

 protected:
  MaceStatus ResizeOutAndPadOut(const OpContext *context,
                                const Tensor *input,
                                const Tensor *filter,
                                const Tensor *output_shape,
                                Tensor *output,
                                std::vector<int> *out_pad_size,
                                std::unique_ptr<Tensor> *padded_output);

  void UnPadOutput(const Tensor &src,
                   const std::vector<int> &out_pad_size,
                   Tensor *dst);

  DeconvComputeParam PreWorkAndGetDeconvParam(
      const OpContext *context, const Tensor *input, Tensor *out_tensor);
  DepthwiseDeconvComputeParam PreWorkAndGetDepthwiseDeconvParam(
      const OpContext *context, const Tensor *input, Tensor *out_tensor);
  GroupDeconvComputeParam PreWorkAndGetGroupDeconvParam(
      const OpContext *context, const Tensor *input, Tensor *out_tensor);

 protected:
  index_t group_;

 private:
  int type_size_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DECONV_2D_H_

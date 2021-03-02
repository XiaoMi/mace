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

#ifndef MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_MXN_H_
#define MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_MXN_H_

#include <vector>
#include <memory>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/ops/arm/base/deconv_2d.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/depthwise_deconv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class DepthwiseDeconv2dKMxN : public Deconv2dBase {
 public:
  explicit DepthwiseDeconv2dKMxN(
      const delegator::DepthwiseDeconv2dParam &param)
      : Deconv2dBase(param, sizeof(T)) {}
  virtual ~DepthwiseDeconv2dKMxN() {}

  MaceStatus Compute(
      const OpContext *context, const Tensor *input, const Tensor *filter,
      const Tensor *output_shape, Tensor *output) override {
    std::unique_ptr<Tensor> padded_out;
    std::vector<int> out_pad_size;
    group_ = input->dim(1);
    ResizeOutAndPadOut(context,
                       input,
                       filter,
                       output_shape,
                       output,
                       &out_pad_size,
                       &padded_out);

    Tensor *out_tensor = output;
    if (padded_out != nullptr) {
      out_tensor = padded_out.get();
    }
    out_tensor->Clear();

    const T *input_data = input->data<float>();
    const T *filter_data = filter->data<float>();
    T *padded_out_data = out_tensor->mutable_data<float>();

    DepthwiseDeconvComputeParam p =
        PreWorkAndGetDepthwiseDeconvParam(context, input, out_tensor);
    DoCompute(p, filter_data, input_data, padded_out_data);
    UnPadOutput(*out_tensor, out_pad_size, output);

    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus DoCompute(
      const DepthwiseDeconvComputeParam &p, const T *filter,
      const T *input_data, T *padded_out_data) = 0;
};

template<typename T>
class GroupDeconv2dKMxN : public Deconv2dBase {
 public:
  explicit GroupDeconv2dKMxN(
      const delegator::DepthwiseDeconv2dParam &param)
      : Deconv2dBase(param, sizeof(T)) {}
  virtual ~GroupDeconv2dKMxN() {}

  MaceStatus Compute(
      const OpContext *context, const Tensor *input, const Tensor *filter,
      const Tensor *output_shape, Tensor *output) override {
    std::unique_ptr<Tensor> padded_out;
    std::vector<int> out_pad_size;
    ResizeOutAndPadOut(context,
                       input,
                       filter,
                       output_shape,
                       output,
                       &out_pad_size,
                       &padded_out);

    Tensor *out_tensor = output;
    if (padded_out != nullptr) {
      out_tensor = padded_out.get();
    }
    out_tensor->Clear();

    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto padded_out_data = out_tensor->mutable_data<float>();

    GroupDeconvComputeParam p =
        PreWorkAndGetGroupDeconvParam(context, input, out_tensor);
    DoCompute(p, filter_data, input_data, padded_out_data);
    UnPadOutput(*out_tensor, out_pad_size, output);

    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus DoCompute(
      const GroupDeconvComputeParam &p, const T *filter,
      const T *input_data, T *padded_out_data) = 0;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_MXN_H_

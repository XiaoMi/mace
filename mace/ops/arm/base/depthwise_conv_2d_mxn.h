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

#ifndef MACE_OPS_ARM_BASE_DEPTHWISE_CONV_2D_MXN_H_
#define MACE_OPS_ARM_BASE_DEPTHWISE_CONV_2D_MXN_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/conv_2d.h"
#include "mace/ops/delegator/depthwise_conv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class DepthwiseConv2dKMxN : public Conv2dBase {
 public:
  explicit DepthwiseConv2dKMxN(const delegator::DepthwiseConv2dParam &param)
      : Conv2dBase(param, sizeof(T)) {}
  virtual ~DepthwiseConv2dKMxN() {}

  MaceStatus Compute(const OpContext *context, const Tensor *input,
                     const Tensor *filter, Tensor *output) {
    DepthwiseConvComputeParam p =
        PreWorkAndGetDepthwiseConv2DParam(context, input, filter, output);

    Tensor::MappingGuard in_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard out_guard(output);
    const T *filter_data = filter->data<T>();
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    DoCompute(p, filter_data, input_data, output_data);

    return MaceStatus::MACE_SUCCESS;
  }

 protected:
  virtual MaceStatus DoCompute(
      const DepthwiseConvComputeParam &p, const T *filter,
      const T *input_data, T *output_data) = 0;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DEPTHWISE_CONV_2D_MXN_H_

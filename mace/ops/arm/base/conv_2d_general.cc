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

#include "mace/ops/arm/base/conv_2d_general.h"

#include <memory>

namespace mace {
namespace ops {
namespace arm {

template<typename T>
MaceStatus Conv2dGeneral<T>::Compute(const OpContext *context,
                                     const Tensor *input,
                                     const Tensor *filter,
                                     Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context, input, filter, output, 1, 4,
                       &padded_input, &padded_output);
  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  Tensor::MappingGuard in_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard out_guard(output);

  const T *filter_data = filter->data<T>();
  const T *input_data = in_tensor->data<T>();
  T *output_data = out_tensor->mutable_data<T>();

  const ConvComputeParam p =
      PreWorkAndGetConv2DParam(context, in_tensor, out_tensor);
  auto &filter_shape = filter->shape();

  DoCompute(p, filter_data, input_data, output_data, filter_shape);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

void RegisterConv2dGeneralDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Conv2dGeneral<float>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY(Conv2d, DeviceType::CPU, float, ImplType::NEON));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace

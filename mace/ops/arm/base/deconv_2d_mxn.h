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

#ifndef MACE_OPS_ARM_BASE_DECONV_2D_MXN_H_
#define MACE_OPS_ARM_BASE_DECONV_2D_MXN_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/deconv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Deconv2dKMxN : public Deconv2dBase {
 public:
  explicit Deconv2dKMxN(const delegator::Deconv2dParam &param)
      : Deconv2dBase(param, sizeof(T)) {}
  virtual ~Deconv2dKMxN() {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *filter,
                     const Tensor *output_shape,
                     Tensor *output) {
    std::unique_ptr<Tensor> padded_out;
    std::vector<int> out_pad_size;
    ResizeOutAndPadOut(context, input, filter, output_shape,
                       output, &out_pad_size, &padded_out);

    Tensor *out_tensor = output;
    if (padded_out != nullptr) {
      out_tensor = padded_out.get();
    }
    out_tensor->Clear();

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard output_mapper(output);

    const T *input_data = input->data<T>();
    const T *filter_data = filter->data<T>();
    T *padded_out_data = out_tensor->mutable_data<T>();

    const DeconvComputeParam p =
        PreWorkAndGetDeconvParam(context, input, out_tensor);
    DoCompute(p, filter_data, input_data, padded_out_data);
    UnPadOutput(*out_tensor, out_pad_size, output);

    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus DoCompute(const DeconvComputeParam &p, const T *filter,
                               const T *input_data, T *padded_out_data) = 0;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DECONV_2D_MXN_H_

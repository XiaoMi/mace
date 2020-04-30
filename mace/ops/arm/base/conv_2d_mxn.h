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

#ifndef MACE_OPS_ARM_BASE_CONV_2D_MXN_H_
#define MACE_OPS_ARM_BASE_CONV_2D_MXN_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/conv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Conv2dKMxN : public Conv2dBase {
 public:
  explicit Conv2dKMxN(const delegator::Conv2dParam &param,
                      const int tile_h, const int tile_w)
      : Conv2dBase(param, sizeof(T)),
        out_tile_h_(tile_h), out_tile_w_(tile_w) {}

  virtual ~Conv2dKMxN() {}

  MaceStatus Compute(const OpContext *context, const Tensor *input,
                     const Tensor *filter, Tensor *output) override {
    std::unique_ptr<const Tensor> padded_input;
    std::unique_ptr<Tensor> padded_output;
    ResizeOutAndPadInOut(context, input, filter, output, out_tile_h_,
                         out_tile_w_, &padded_input, &padded_output);
    const Tensor *in_tensor = input;
    if (padded_input != nullptr) {
      in_tensor = padded_input.get();
    }
    Tensor *out_tensor = output;
    if (padded_output != nullptr) {
      out_tensor = padded_output.get();
    }
    out_tensor->Clear();

    const T *filter_data = filter->data<T>();
    const T *input_data = in_tensor->data<T>();
    T *output_data = out_tensor->mutable_data<T>();

    const ConvComputeParam p =
        PreWorkAndGetConv2DParam(context, in_tensor, out_tensor);

    DoCompute(p, filter_data, input_data, output_data);

    UnPadOutput(*out_tensor, output);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus DoCompute(const ConvComputeParam &p, const T *filter,
                               const T *input_data, T *output_data) = 0;

 private:
  const int out_tile_h_;
  const int out_tile_w_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_CONV_2D_MXN_H_

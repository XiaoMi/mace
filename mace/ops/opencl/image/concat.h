// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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
#ifndef MACE_OPS_OPENCL_IMAGE_CONCAT_H_
#define MACE_OPS_OPENCL_IMAGE_CONCAT_H_

#include "mace/ops/opencl/concat.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace concat {
MaceStatus Concat2(OpContext *context,
                   cl::Kernel *kernel,
                   const Tensor *input0,
                   const Tensor *input1,
                   const DataType dt,
                   std::vector<index_t> *prev_input_shape,
                   Tensor *output,
                   uint32_t *kwg_size);

MaceStatus ConcatN(OpContext *context,
                   cl::Kernel *kernel,
                   const std::vector<const Tensor *> &input_list,
                   const DataType dt,
                   Tensor *output,
                   uint32_t *kwg_size);
}  // namespace concat

template <typename T>
class ConcatKernel : public OpenCLConcatKernel {
 public:
  explicit ConcatKernel(const int32_t axis) : axis_(axis) {}
  MaceStatus Compute(
      OpContext *context,
      const std::vector<const Tensor *> &input_list,
      Tensor *output) override;

 private:
  int32_t axis_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus ConcatKernel<T>::Compute(
    OpContext *context,
    const std::vector<const Tensor *> &input_list,
    Tensor *output) {
  const int inputs_count = input_list.size();
  MACE_CHECK(inputs_count >= 2 && axis_ == 3)
    << "Concat opencl kernel only support >=2 elements with axis == 3";

  const Tensor *input0 = input_list[0];
  bool divisible_four = input0->dim(axis_) % 4 == 0;

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    divisible_four &= input->dim(axis_) % 4 == 0;
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis_) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis_] += input->dim(axis_);
  }
  MACE_CHECK(
      inputs_count == 2 || divisible_four,
      "Dimensions of inputs should be divisible by 4 when inputs_count > 2.");
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  switch (inputs_count) {
    case 2:
      return concat::Concat2(
          context, &kernel_, input_list[0], input_list[1],
          DataTypeToEnum<T>::value, &input_shape_, output, &kwg_size_);
    default:
      if (divisible_four) {
        return concat::ConcatN(context, &kernel_, input_list,
                               DataTypeToEnum<T>::value, output, &kwg_size_);
      } else {
        MACE_NOT_IMPLEMENTED;
      }
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONCAT_H_

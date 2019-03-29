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
  ConcatKernel() {}
  MaceStatus Compute(
      OpContext *context,
      const std::vector<const Tensor *> &input_list,
      const int32_t axis,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus ConcatKernel<T>::Compute(
    OpContext *context,
    const std::vector<const Tensor *> &input_list,
    const int32_t axis,
    Tensor *output) {
  const int inputs_count = input_list.size();

  const Tensor *input0 = input_list[0];

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis] += input->dim(axis);
  }
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  switch (inputs_count) {
    case 2:
      return concat::Concat2(
          context, &kernel_, input_list[0], input_list[1],
          DataTypeToEnum<T>::value, &input_shape_, output, &kwg_size_);
    default:
      return concat::ConcatN(context, &kernel_, input_list,
                             DataTypeToEnum<T>::value, output, &kwg_size_);
  }
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONCAT_H_

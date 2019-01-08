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

#ifndef MACE_OPS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_
#define MACE_OPS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_

#include "mace/ops/opencl/buffer_transform_kernel.h"

#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus BufferTypeTransform(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output);

MaceStatus TransformConv2DFilter(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output);

MaceStatus TransformDWConv2DFilter(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output);

MaceStatus TransformArgument(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output);


template <typename T>
class BufferTransform: public OpenCLBufferTransformKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const OpenCLBufferType type,
      const int wino_blk_size,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus BufferTransform<T>::Compute(OpContext *context,
                                       const Tensor *input,
                                       const OpenCLBufferType type,
                                       const int wino_blk_size,
                                       Tensor *output) {
  MACE_UNUSED(wino_blk_size);
  const DataType dt = DataTypeToEnum<T>::value;
  switch (type) {
    case CONV2D_FILTER:
      return TransformConv2DFilter(context, &kernel_, input, dt, output);
    case DW_CONV2D_FILTER:
      return TransformDWConv2DFilter(context, &kernel_, input, dt, output);
    case ARGUMENT:
      return TransformArgument(context, &kernel_, input, dt, output);
    default:
      if (input->dtype() != dt) {
        return BufferTypeTransform(context, &kernel_, input, dt, output);
      } else {
        SetFutureDefaultWaitFn(context->future());
        output->ReuseTensorBuffer(*input);
        return MaceStatus::MACE_SUCCESS;
      }
  }
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_

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

#ifndef MACE_KERNELS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_
#define MACE_KERNELS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_

#include <vector>

#include "mace/kernels/buffer_transform.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {
namespace opencl {
namespace buffer {

MaceStatus BufferTypeTransform(
    OpKernelContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output,
    StatsFuture *future);

MaceStatus TransformConv2DFilter(
    OpKernelContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output,
    StatsFuture *future);

MaceStatus TransformDWConv2DFilter(
    OpKernelContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output,
    StatsFuture *future);

MaceStatus TransformArgument(
    OpKernelContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    const DataType dt,
    Tensor *output,
    StatsFuture *future);


template <typename T>
class BufferTransform: public OpenCLBufferTransformKernel {
 public:
  MaceStatus Compute(
      OpKernelContext *context,
      const Tensor *input,
      const BufferType type,
      const int wino_blk_size,
      Tensor *output,
      StatsFuture *future) override;

 private:
  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus BufferTransform<T>::Compute(OpKernelContext *context,
                                       const Tensor *input,
                                       const BufferType type,
                                       const int wino_blk_size,
                                       Tensor *output,
                                       StatsFuture *future) {
  MACE_UNUSED(type);
  MACE_UNUSED(wino_blk_size);
  const DataType dt = DataTypeToEnum<T>::value;
  switch (type) {
    case CONV2D_FILTER:
      return TransformConv2DFilter(context, &kernel_, input,
                                   dt, output, future);
    case DW_CONV2D_FILTER:
      return TransformDWConv2DFilter(context, &kernel_, input,
                                     dt, output, future);
    case ARGUMENT:
      return TransformArgument(context, &kernel_, input, dt, output, future);
    default:
      if (input->dtype() != dt) {
        return BufferTypeTransform(context, &kernel_, input,
                                   dt, output, future);
      } else {
        SetFutureDefaultWaitFn(future);
        output->ReuseTensorBuffer(*input);
        return MaceStatus::MACE_SUCCESS;
      }
  }
}

}  // namespace buffer
}  // namespace opencl
}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_OPENCL_BUFFER_BUFFER_TRANSFORM_H_

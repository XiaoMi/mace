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

#ifndef MACE_KERNELS_SQRDIFF_MEAN_H_
#define MACE_KERNELS_SQRDIFF_MEAN_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct SqrDiffMeanFunctor : OpKernel {
  explicit SqrDiffMeanFunctor(OpKernelContext *context)
  : OpKernel(context) {}

  void Compute(const Tensor *input0,
               const Tensor *input1,
               Tensor *output) {
    Tensor::MappingGuard input0_mapper(input0);
    Tensor::MappingGuard input1_mapper(input1);
    const T *input_ptr0 = input0->data<T>();
    const T *input_ptr1 = input1->data<T>();
    Tensor::MappingGuard output_map(output);
    T *output_ptr = output->mutable_data<T>();
    memset(output_ptr, 0, output->size() * sizeof(T));

    const index_t img_size = input0->dim(2) * input0->dim(3);
    const index_t bc = input0->dim(0) * input0->dim(1);
#pragma omp parallel for
    for (int i = 0; i < bc; ++i) {
      for (int j = 0; j < img_size; ++j) {
        T diff = input_ptr0[i * img_size + j] - input_ptr1[i];
        output_ptr[i] += diff * diff;
      }
      output_ptr[i] /= img_size;
    }
  }

  MaceStatus operator()(const Tensor *input0,
                        const Tensor *input1,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    MACE_CHECK(input0->dim(0) == input1->dim(0) &&
        input0->dim(1) == input1->dim(1),
               "inputs dims N and C should be the same.");

    std::vector<index_t> out_shape(4);
    out_shape[0] = input0->dim(0);
    out_shape[1] = input0->dim(1);
    out_shape[2] = 1;
    out_shape[3] = 1;

    output->Resize(out_shape);
    Compute(input0, input1, output);
    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
class OpenCLSqrDiffMeanKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const Tensor *input0,
      const Tensor *input1,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLSqrDiffMeanKernel);
};
template <typename T>
struct SqrDiffMeanFunctor<DeviceType::GPU, T> : OpKernel {
  explicit SqrDiffMeanFunctor(OpKernelContext *context);

  MaceStatus operator()(const Tensor *input0,
                        const Tensor *input1,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLSqrDiffMeanKernel> kernel_;
};
#endif

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SQRDIFF_MEAN_H_

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

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/sqrdiff_mean.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SqrDiffMeanOp : public Operation {
 public:
  explicit SqrDiffMeanOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->Input(1);
    Tensor *output = this->Output(0);

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
    return MaceStatus::MACE_SUCCESS;
  }

 private:
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

    for (int i = 0; i < bc; ++i) {
      for (int j = 0; j < img_size; ++j) {
        T diff = input_ptr0[i * img_size + j] - input_ptr1[i];
        output_ptr[i] += diff * diff;
      }
      output_ptr[i] /= img_size;
    }
  }
};


#ifdef MACE_ENABLE_OPENCL
template <typename T>
class SqrDiffMeanOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit SqrDiffMeanOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SqrDiffMeanKernel<T>>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->Input(1);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input0, input1, output);
  }

 private:
  std::unique_ptr<OpenCLSqrDiffMeanKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterSqrDiffMean(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "SqrDiffMean", SqrDiffMeanOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "SqrDiffMean", SqrDiffMeanOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "SqrDiffMean", SqrDiffMeanOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

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

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <memory>

#include "mace/core/operator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/kernels/opencl/image/addn.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

static constexpr int kCostPerGroup = 1024;

template <DeviceType D, class T>
class AddNOp;

template <>
class AddNOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit AddNOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Tensor *output_tensor = this->Output(0);
    size_t input_size = this->inputs_.size();
    MACE_RETURN_IF_ERROR(output_tensor->ResizeLike(inputs_[0]));
    index_t size = output_tensor->size();
    Tensor::MappingGuard output_map(output_tensor);
    float *output_data = output_tensor->mutable_data<float>();
    memset(output_data, 0, size * sizeof(float));
    int64_t cost = size * input_size;
    int64_t groups = 1;
    if (cost > kCostPerGroup) {
      groups = cost / kCostPerGroup;
    }
    int64_t element_per_group = size / groups;

    std::vector<Tensor::MappingGuard> mappers;
    for (size_t i = 0; i < input_size; ++i) {
      MACE_CHECK(inputs_[0]->dim_size() == inputs_[i]->dim_size());
      MACE_CHECK(inputs_[0]->size() == inputs_[i]->size())
        << "Input 0: " << MakeString(inputs_[0]->shape())
        << ", size: " << inputs_[0]->size() << ". Input " << i << ": "
        << MakeString(inputs_[i]->shape()) << ", size: " << inputs_[i]->size();
      mappers.emplace_back(Tensor::MappingGuard(inputs_[i]));
    }

#pragma omp parallel for
    for (int64_t i = 0; i < size; i += element_per_group) {
      int64_t count = std::min(element_per_group, size - i);
      int nn = count >> 2;
      int remain = count - (nn << 2);
      for (size_t j = 0; j < input_size; ++j) {
        const float *input_data = inputs_[j]->data<float>();
        const float *input_ptr = input_data + i;
        float *output_ptr = output_data + i;
        for (int k = 0; k < nn; ++k) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
          float32x4_t in = vld1q_f32(input_ptr);
          float32x4_t out = vld1q_f32(output_ptr);
          out = vaddq_f32(out, in);
          vst1q_f32(output_ptr, out);
#else
          for (int m = 0; m < 4; ++m) {
            output_ptr[m] += input_ptr[m];
          }
#endif

          input_ptr += 4;
          output_ptr += 4;
        }
        for (int k = 0; k < remain; ++k) {
          *output_ptr += *input_ptr;
          ++input_ptr;
          ++output_ptr;
        }
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class AddNOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit AddNOp(OpConstructContext *context)
      : Operation(context) {
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::AddNKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    Tensor *output_tensor = this->Output(0);
    size_t n = this->inputs_.size();
    for (size_t i = 1; i < n; ++i) {
      MACE_CHECK(inputs_[0]->dim_size() == inputs_[i]->dim_size());
      MACE_CHECK(inputs_[0]->size() == inputs_[i]->size())
        << "Input 0: " << MakeString(inputs_[0]->shape())
        << ", size: " << inputs_[0]->size() << ". Input " << i << ": "
        << MakeString(inputs_[i]->shape()) << ", size: " << inputs_[i]->size();
    }

    return kernel_->Compute(context, inputs_, output_tensor);
  }

 private:
  std::unique_ptr<OpenCLAddNKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterAddN(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "AddN", AddNOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace kernels
}  // namespace mace

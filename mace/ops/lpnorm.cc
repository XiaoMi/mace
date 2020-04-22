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


#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/lpnorm.h"
#endif  // MACE_ENABLE_OPENCL

/**
 * LpNormOp is a Normalization OP which support L1 and L2, which is a custom op
 * of caffe (not exist in official caffe), please reference:
 * https://github.com/freesouls/caffe/blob/master/src/caffe/layers/normalization_layer.cpp  #noqa
 */

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class LpNormOp;

template<class T>
class LpNormOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit LpNormOp(OpConstructContext *context)
      : Operation(context),
        p_(Operation::GetOptionalArg<int>("p", 2)),
        axis_(Operation::GetOptionalArg<int>("axis", -1)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    if (axis_ < 0) {
      axis_ += input->dim_size();
    }
    MACE_CHECK(axis_ < input->dim_size() && axis_ >= 0,
               "The axis_ must be small than dim size");
    const std::vector<index_t> &input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->Resize(input_shape));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);

    const auto *input_data = input->data<T>();
    auto *output_data = output->mutable_data<T>();
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    auto outer_loop = std::accumulate(input_shape.begin(),
                                      input_shape.begin() + axis_, 1,
                                      std::multiplies<index_t>());
    auto inner_loop = std::accumulate(input_shape.begin() + axis_,
                                      input_shape.end(), 1,
                                      std::multiplies<index_t>());

    if (p_ == 1) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = std::abs(input_data[i]);
        }
      }, 0, input->size(), 1);
    } else if (p_ == 2) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = input_data[i] * input_data[i];
        }
      }, 0, input->size(), 1);
    } else {
      LOG(FATAL) << "LpNorm's p should be 1 or 2, current p is: " << p_;
    }

    const float power = 1 / static_cast<float>(p_);
    auto norm_buffer = context->device()->scratch_buffer();
    norm_buffer->Rewind();
    MACE_RETURN_IF_ERROR(norm_buffer->GrowSize(outer_loop * sizeof(float)));
    float *norm_ptr = norm_buffer->mutable_data<float>();
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        auto output_data_base = output_data + inner_loop * i;
        norm_ptr[i] = std::accumulate(output_data_base,
                                      output_data_base + inner_loop,
                                      static_cast<T>(0.0f));
        norm_ptr[i] = std::pow(norm_ptr[i], power);
        norm_ptr[i] += 1e-6;
      }
    }, 0, outer_loop, 1);

    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        const auto offset = i * inner_loop;
        for (index_t j = start1; j < end1; j += step1) {
          output_data[offset + j] = input_data[offset + j] / norm_ptr[i];
        }
      }
    }, 0, outer_loop, 1, 0, inner_loop, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int p_;
  int axis_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class LpNormOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit LpNormOp(OpConstructContext *context)
      : Operation(context) {
    const auto p = Operation::GetOptionalArg<int>("p", 2);
    const auto axis = Operation::GetOptionalArg<int>("axis", -1);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::LpNormKernel>(p, axis);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLLpNormKernel> kernel_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterLpNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "LpNorm", LpNormOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "LpNorm", LpNormOp,
                        DeviceType::CPU);
  MACE_REGISTER_GPU_OP(op_registry, "LpNorm", LpNormOp);
}

}  // namespace ops
}  // namespace mace

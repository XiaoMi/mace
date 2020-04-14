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
#include "mace/ops/opencl/image/mvnorm.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

// Mean-Variance Normalization (MVN)
template<DeviceType D, typename T>
class MVNormOp;

template<class T>
class MVNormOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit MVNormOp(OpConstructContext *context)
      : Operation(context),
        normalize_variance_(
            Operation::GetOptionalArg<bool>("normalize_variance", true)),
        across_channels_(
            Operation::GetOptionalArg<bool>("across_channels", false)),
        eps_(Operation::GetOptionalArg<float>("epsilon", 1e-9)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->data_format() == DataFormat::NCHW,
               "The MVN only suport NCHW");
    const std::vector<index_t> &input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->Resize(input_shape));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);
    const auto *input_data = input->data<T>();
    auto *output_data = output->mutable_data<T>();

    const auto input_size = input->size();
    const auto outer_loop =
        across_channels_ ? input_shape[0] : input_shape[0] * input_shape[1];
    const auto inner_loop = input_size / outer_loop;
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    Buffer mean_buffer(context->device()->allocator());
    MACE_RETURN_IF_ERROR(mean_buffer.Allocate(outer_loop * sizeof(float)));
    auto *mean_ptr = mean_buffer.mutable_data<float>();

    // compute EX
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        const auto offset = inner_loop * i;
        mean_ptr[i] = std::accumulate(input_data + offset,
                                      input_data + offset + inner_loop,
                                      static_cast<T>(0.0f));
        mean_ptr[i] /= inner_loop;
      }
    }, 0, outer_loop, 1);

    // compute (X - EX)
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        const auto offset = i * inner_loop;
        for (index_t j = start1; j < end1; j += step1) {
          output_data[offset + j] = input_data[offset + j] - mean_ptr[i];
        }
      }
    }, 0, outer_loop, 1, 0, inner_loop, 1);

    if (normalize_variance_) {
      // compute (X - EX)^2
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = output_data[i] * output_data[i];
        }
      }, 0, input_size, 1);

      auto mean_v_buffer = context->device()->scratch_buffer();
      mean_v_buffer->Rewind();
      MACE_RETURN_IF_ERROR(
          mean_v_buffer->GrowSize(outer_loop * sizeof(float)));
      float *mean_v_ptr = mean_v_buffer->mutable_data<float>();
      // compute E((X - EX)^2)^0.5 + eps_
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          auto output_data_base = output_data + inner_loop * i;
          mean_v_ptr[i] = std::accumulate(output_data_base,
                                          output_data_base + inner_loop,
                                          static_cast<T>(0.0f));
          mean_v_ptr[i] = std::pow(mean_v_ptr[i] / inner_loop, 0.5f) + eps_;
        }
      }, 0, outer_loop, 1);

      // compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          const auto offset = i * inner_loop;
          for (index_t j = start1; j < end1; j += step1) {
            output_data[offset + j] =
                (input_data[offset + j] - mean_ptr[i]) / mean_v_ptr[i];
          }
        }
      }, 0, outer_loop, 1, 0, inner_loop, 1);
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool normalize_variance_;
  bool across_channels_;
  float eps_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class MVNormOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit MVNormOp(OpConstructContext *context) : Operation(context) {
    auto normalize_variance =
        Operation::GetOptionalArg<bool>("normalize_variance", true);
    auto across_channels =
        Operation::GetOptionalArg<bool>("across_channels", false);
    auto eps = Operation::GetOptionalArg<float>("epsilon", 1e-9);

    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::MVNormKernel>(
          normalize_variance, across_channels, eps);
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
  std::unique_ptr<OpenCLMVNormKernel> kernel_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterMVNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "MVNorm", MVNormOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "MVNorm", MVNormOp,
                        DeviceType::CPU);
  MACE_REGISTER_GPU_OP(op_registry, "MVNorm", MVNormOp);
}

}  // namespace ops
}  // namespace mace

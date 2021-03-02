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

#include <memory>
#include <string>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/activation.h"
#include "mace/ops/delegator/activation.h"
#include "mace/utils/memory.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/mvnorm.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class GroupNormOp;

template<class T>
class GroupNormOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit GroupNormOp(OpConstructContext *context)
      : Operation(context),
        eps_(Operation::GetOptionalArg<float>("epsilon",
                                              static_cast<float>(1e-5))),
        group_num_(Operation::GetOptionalArg<int>("group_num", 32)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = Input(INPUT);
    Tensor *output = Output(OUTPUT);
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    const std::vector<index_t> &input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->Resize(input_shape));

    const auto batch = input_shape[0];
    const auto channel = input_shape[1];
    const auto height = input_shape[2];
    const auto width = input_shape[3];
    MACE_CHECK(channel % group_num_ == 0,
               "group_num_ invalid.", channel, group_num_);
    const auto group_size = channel / group_num_;

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    const auto outer_loop = batch * group_num_;
    const auto inner_loop = group_size * height * width;
    utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

    auto *runtime = context->runtime();
    MemInfo mem_info(input->memory_type(),
                     DataType::DT_FLOAT, {outer_loop * 2});
    auto scratch_buffer = runtime->ObtainBuffer(mem_info, RENT_SCRATCH);
    float *mean_ptr = scratch_buffer->mutable_data<float>();
    float *variance_ptr = mean_ptr + outer_loop;

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

    // compute (X - EX)^2
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        const auto offset = i * inner_loop;
        for (index_t j = start1; j < end1; j += step1) {
          const auto idx = offset + j;
          const auto x_ex = input_data[idx] - mean_ptr[i];
          output_data[idx] = x_ex * x_ex;
        }
      }
    }, 0, outer_loop, 1, 0, inner_loop, 1);

    // compute (E((X - EX)^2) + eps_)^0.5
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        auto output_data_base = output_data + inner_loop * i;
        variance_ptr[i] = std::accumulate(output_data_base,
                                          output_data_base + inner_loop,
                                          static_cast<T>(0.0f));
        variance_ptr[i] = std::pow(variance_ptr[i] / inner_loop + eps_, 0.5f);
      }
    }, 0, outer_loop, 1);

    // compute (X - EX) / ((E((X - EX)^2) + eps_)^0.5)
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        const auto offset = i * inner_loop;
        for (index_t j = start1; j < end1; j += step1) {
          output_data[offset + j] =
              (input_data[offset + j] - mean_ptr[i]) / variance_ptr[i];
        }
      }
    }, 0, outer_loop, 1, 0, inner_loop, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  const float eps_;
  const int group_num_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class GroupNormOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit GroupNormOp(OpConstructContext *context) : Operation(context) {
    const auto group_num = Operation::GetOptionalArg<int>("group_num", 32);
    const auto eps = Operation::GetOptionalArg<float>(
        "epsilon", static_cast<float>(1e-5));
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::MVNormKernel>(
          true, opencl::image::MeanType::GROUP_CHANNELS, eps, group_num);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLMVNormKernel> kernel_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterGroupNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "GroupNorm", GroupNormOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "GroupNorm",
                        GroupNormOp, RuntimeType::RT_CPU);
  MACE_REGISTER_GPU_OP(op_registry, "GroupNorm", GroupNormOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry, OpConditionBuilder("GroupNorm").SetDevicePlacerFunc(
      [](OpConditionContext *context) -> std::set<RuntimeType> {
        auto op = context->operator_def();
        if (op->output_shape_size() != op->output_size()) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        const int group_num = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *op, "group_num", 32);
        auto output_channels = op->output_shape(0).dims()[3];
        const int group_size = output_channels / group_num;
        if (group_size % 4 == 0) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        return {RuntimeType::RT_CPU};
      }));
}

}  // namespace ops
}  // namespace mace

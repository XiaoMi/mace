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

#include "mace/ops/activation.h"

#include <memory>
#include <set>

#include "mace/core/operator.h"

#if defined(MACE_ENABLE_NEON)
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/activation.h"
#endif

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/activation.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ActivationOp;

template<>
class ActivationOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit ActivationOp(OpConstructContext *context)
      : Operation(context),
        activation_type_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        activation_delegator_(activation_type_,
                              Operation::GetOptionalArg<float>("max_limit",
                                                               0.0f),
                              Operation::GetOptionalArg<float>(
                                  "leakyrelu_coefficient", 0.0f)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    if (activation_type_ == PRELU) {
      MACE_RETURN_IF_ERROR(output->ResizeLike(input));
      const float *input_ptr = input->data<float>();
      float *output_ptr = output->mutable_data<float>();
      MACE_CHECK(this->InputSize() > 1);
      const Tensor *alpha = this->Input(1);
      const float *alpha_ptr = alpha->data<float>();
      const index_t outer_size = output->dim(0);
      const index_t inner_size = output->dim(2) * output->dim(3);
      PReLUActivation(context, input_ptr, outer_size, input->dim(1), inner_size,
                      alpha_ptr, output_ptr);
    } else {
      activation_delegator_.Compute(context, input, output);
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  ActivationType activation_type_;
#if defined(MACE_ENABLE_NEON)
  arm::fp32::Activation activation_delegator_;
#else
  ref::Activation activation_delegator_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ActivationOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit ActivationOp(OpConstructContext *context)
      : Operation(context) {
    ActivationType type = ops::StringToActivationType(
        Operation::GetOptionalArg<std::string>("activation",
                                              "NOOP"));
    auto relux_max_limit = static_cast<T>(
        Operation::GetOptionalArg<float>("max_limit", 0.0f));
    auto leakyrelu_coefficient = static_cast<T>(
        Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f));
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::ActivationKernel<T>>(
          type, relux_max_limit, leakyrelu_coefficient);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    if (type == ActivationType::PRELU) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 1, OpenCLBufferType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *alpha = this->InputSize() > 1 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return kernel_->Compute(context, input, alpha, output);
  }

 private:
  std::unique_ptr<OpenCLActivationKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterActivation(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Activation", ActivationOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Activation", ActivationOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Activation", ActivationOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Activation")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                int has_data_format =
                    ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                        *op, "has_data_format", 0);
                if (!has_data_format ||
                    op->output_shape(0).dims_size() != 4) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace

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
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/delegator/activation.h"
#include "mace/ops/delegator/gemv.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/fully_connected.h"
#include "mace/runtimes/opencl/transform/buffer_transformer.h"
#endif  // MACE_ENABLE_OPENCL

#include "mace/runtimes/cpu/cpu_runtime.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class FullyConnectedOpBase : public Operation {
 public:
  explicit FullyConnectedOpBase(OpConstructContext *context)
      : Operation(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        activation_coefficient_(Operation::GetOptionalArg<float>(
            "activation_coefficient", 0.0f)) {}
 protected:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float activation_coefficient_;

  MACE_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

template<RuntimeType D, class T>
class FullyConnectedOp;

template<class T>
class FullyConnectedOp<RuntimeType::RT_CPU, T> : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context),
        activation_delegator_(delegator::Activation::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(Activation,
                               RuntimeType::RT_CPU, T, kCpuImplType),
            delegator::ActivationParam(
                activation_,
                relux_max_limit_,
                activation_coefficient_,
                Operation::GetOptionalArg<float>(
                    "hardsigmoid_alpha", 0.f),
                Operation::GetOptionalArg<float>(
                    "hardsigmoid_beta", 0.f)))),
        gemv_(delegator::Gemv::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(Gemv, RuntimeType::RT_CPU, T, kCpuImplType),
            DelegatorParam())) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(
        input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
            input->dim(3) == weight->dim(3),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }
    std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const index_t batch = output->dim(0);
    const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
    const index_t output_size = weight->dim(0);

    gemv_->Compute(context,
                   weight,
                   input,
                   bias,
                   batch,
                   output_size,
                   input_size,
                   false,
                   true,
                   output);

    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::unique_ptr<delegator::Activation> activation_delegator_;
  std::unique_ptr<delegator::Gemv> gemv_;
};

#ifdef MACE_ENABLE_QUANTIZE
template<>
class FullyConnectedOp<RuntimeType::RT_CPU, uint8_t>
    : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context),
        gemv_(delegator::Gemv::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(Gemv, RuntimeType::RT_CPU,
                               uint8_t, kCpuImplType),
            DelegatorParam())) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(
        input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
            input->dim(3) == weight->dim(3),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }
    auto gemm_context = CpuRuntime::Get(context)->GetGemmlowpContext();
    MACE_CHECK_NOTNULL(gemm_context);

    std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const int batch = static_cast<int>(output->dim(0));
    const int input_size =
        static_cast<int>(weight->dim(1) * weight->dim(2) * weight->dim(3));
    const int output_size = static_cast<int>(weight->dim(0));
    gemv_->Compute(context,
                  weight,
                  input,
                  bias,
                  batch,
                  output_size,
                  input_size,
                  false,
                  true,
                  output);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::unique_ptr<delegator::Gemv> gemv_;
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template<>
class FullyConnectedOp<RuntimeType::RT_OPENCL, float>
    : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context) {
    MemoryType mem_type = MemoryType::CPU_BUFFER;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::FullyConnectedKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filter tensor to target format
    MACE_CHECK(TransformFilter(
        context,
        operator_def_.get(),
        1,
        BufferContentType::WEIGHT_WIDTH,
        mem_type) == MaceStatus::MACE_SUCCESS);
    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter(context, operator_def_.get(), 2,
                                 BufferContentType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(
        input->dim(1) == weight->dim(2) && input->dim(2) == weight->dim(3) &&
            input->dim(3) == weight->dim(1),
        "The shape of Input: ", MakeString(input->shape()),
        "The shape of Weight: ", MakeString(weight->shape()),
        " don't match.");
    return kernel_->Compute(
        context, input, weight, bias, activation_, relux_max_limit_,
        activation_coefficient_, output);
  }

 private:
  std::unique_ptr<OpenCLFullyConnectedKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterFullyConnected(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "FullyConnected",
                        FullyConnectedOp, RuntimeType::RT_CPU);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, RuntimeType::RT_CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

  MACE_REGISTER_GPU_OP(op_registry, "FullyConnected", FullyConnectedOp);
}

}  // namespace ops
}  // namespace mace

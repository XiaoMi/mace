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

#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/gemm.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/gemmlowp_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/fully_connected.h"
#endif  // MACE_ENABLE_OPENCL

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
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)) {}
 protected:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;

  MACE_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

template <DeviceType D, class T>
class FullyConnectedOp;

template <>
class FullyConnectedOp<DeviceType::CPU, float> : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context) {}

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
    const index_t N = output->dim(0);
    const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
    const index_t output_size = weight->dim(0);

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_weight(weight);
    Tensor::MappingGuard guard_output(output);
    const float *input_ptr = input->data<float>();
    const float *weight_ptr = weight->data<float>();
    float *output_ptr = output->mutable_data<float>();

    Gemv(weight_ptr, input_ptr, N, input_size, output_size, output_ptr);

    if (bias) {
      Tensor::MappingGuard guard_bias(bias);
      const float *bias_ptr = bias == nullptr ? nullptr : bias->data<float>();
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < output_size; ++j) {
          output_ptr[j + i * output_size] += bias_ptr[j];
        }
      }
    }

    DoActivation(output_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_, leakyrelu_coefficient_);

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_QUANTIZE
template <>
class FullyConnectedOp<DeviceType::CPU, uint8_t>
    : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context) {}

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
    auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
    MACE_CHECK_NOTNULL(gemm_context);

    std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const int N = static_cast<int>(output->dim(0));
    const int input_size =
        static_cast<int>(weight->dim(1) * weight->dim(2) * weight->dim(3));
    const int output_size = static_cast<int>(weight->dim(0));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_weight(weight);
    Tensor::MappingGuard guard_output(output);
    auto input_ptr = input->data<uint8_t>();
    auto weight_ptr = weight->data<uint8_t>();
    auto output_ptr = output->mutable_data<uint8_t>();

    std::vector<index_t> bias_shape{output_size};
    std::unique_ptr<Tensor> zero_bias;
    const int32_t *bias_ptr = nullptr;
    if (bias == nullptr) {
      zero_bias.reset(
          new Tensor(GetCPUAllocator(), DT_INT32));
      zero_bias->Resize(bias_shape);
      zero_bias->Clear();
      bias_ptr = zero_bias->data<int32_t>();
    } else {
      bias_ptr = bias->data<int32_t>();
    }

    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
        weight_matrix(weight_ptr, output_size, input_size);
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
        input_matrix(input_ptr, input_size, N);
    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::ColMajor>
        output_matrix(output_ptr, output_size, N);

    const auto &output_pipeline = GemmlowpOutputPipeline::Make(
        bias_ptr, output_size, weight->scale(), input->scale(), output->scale(),
        output->zero_point());

    using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
    gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
        gemm_context, weight_matrix, input_matrix, &output_matrix,
        -weight->zero_point(), -input->zero_point(), output_pipeline);

    return MaceStatus::MACE_SUCCESS;
  }
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class FullyConnectedOp<DeviceType::GPU, T> : public FullyConnectedOpBase {
 public:
  explicit FullyConnectedOp(OpConstructContext *context)
      : FullyConnectedOpBase(context) {
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_.reset(new opencl::image::FullyConnectedKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filter tensor to target format
    MACE_CHECK(TransformFilter<T>(
        context,
        operator_def_.get(),
        1,
        OpenCLBufferType::WEIGHT_WIDTH,
        mem_type) == MaceStatus::MACE_SUCCESS);
    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type)
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
        leakyrelu_coefficient_, output);
  }

 private:
  std::unique_ptr<OpenCLFullyConnectedKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterFullyConnected(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

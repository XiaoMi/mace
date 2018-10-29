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

#include "mace/core/operator.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/opencl/image/winograd_transform.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
class WinogradTransformOp;

template <typename T>
class WinogradTransformOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit WinogradTransformOp(OpConstructContext *context)
      : Operation(context) {
    Padding padding_type = static_cast<Padding>(Operation::GetOptionalArg<int>(
        "padding", static_cast<int>(VALID)));
    std::vector<int> paddings = Operation::GetRepeatedArgs<int>(
        "padding_values");
    int block_size = Operation::GetOptionalArg<int>("wino_block_size", 2);
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::WinogradTransformKernel<T>(
          padding_type, paddings, block_size));
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input_tensor = this->Input(0);
    Tensor *output_tensor = this->Output(0);
    return kernel_->Compute(context, input_tensor, output_tensor);
  }

 private:
  std::unique_ptr<OpenCLWinogradTransformKernel> kernel_;
};

template <DeviceType D, typename T>
class WinogradInverseTransformOp;

template <typename T>
class WinogradInverseTransformOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit WinogradInverseTransformOp(OpConstructContext *context)
      : Operation(context) {
    ActivationType activation = kernels::StringToActivationType(
        Operation::GetOptionalArg<std::string>("activation", "NOOP"));
    float relux_max_limit = Operation::GetOptionalArg<float>("max_limit", 0.0f);
    int block_size = Operation::GetOptionalArg<int>("wino_block_size", 2);
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::WinogradInverseTransformKernel<T>(
          activation, relux_max_limit, block_size));
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  MaceStatus Run(OpContext *context) override {
    Tensor *output_tensor = this->Output(0);
    return kernel_->Compute(context, inputs_, output_tensor);
  }

 private:
  std::unique_ptr<OpenCLWinogradInverseTransformKernel> kernel_;
};

void RegisterWinogradTransform(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "WinogradTransform",
                   WinogradTransformOp, DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "WinogradTransform",
                   WinogradTransformOp, DeviceType::GPU, half);
}

void RegisterWinogradInverseTransform(
    OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "WinogradInverseTransform",
                   WinogradInverseTransformOp, DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "WinogradInverseTransform",
                   WinogradInverseTransformOp, DeviceType::GPU, half);
}

}  // namespace kernels
}  // namespace mace

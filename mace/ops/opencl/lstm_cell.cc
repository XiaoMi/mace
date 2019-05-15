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

#include "mace/ops/opencl/lstm_cell.h"

#include <algorithm>
#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/lstm_cell.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class LSTMCellOp;

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class LSTMCellOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit LSTMCellOp(OpConstructContext *context)
      : Operation(context) {
    T forget_bias = static_cast<T>(
        Operation::GetOptionalArg<float>("scalar_input",
                                         0.0));
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::LSTMCellKernel<T>>(forget_bias);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filters
    const Tensor *pre_output = context->workspace()->GetTensor(
        operator_def_->input(1));
    if (pre_output->is_weight()) {
      MACE_CHECK(TransformFilter<T>(context,
                                    operator_def_.get(),
                                    1,
                                    OpenCLBufferType::IN_OUT_CHANNEL,
                                    mem_type) == MaceStatus::MACE_SUCCESS);
    }
    MACE_CHECK(TransformFilter<T>(context,
                                  operator_def_.get(),
                                  2,
                                  OpenCLBufferType::IN_OUT_CHANNEL,
                                  mem_type) == MaceStatus::MACE_SUCCESS);
    MACE_CHECK(TransformFilter<T>(context,
                                  operator_def_.get(),
                                  3,
                                  OpenCLBufferType::ARGUMENT,
                                  mem_type) == MaceStatus::MACE_SUCCESS);
    const Tensor *pre_cell = context->workspace()->GetTensor(
        operator_def_->input(4));
    if (pre_cell->is_weight()) {
      MACE_CHECK(TransformFilter<T>(context,
                                    operator_def_.get(),
                                    4,
                                    OpenCLBufferType::IN_OUT_CHANNEL,
                                    mem_type) == MaceStatus::MACE_SUCCESS);
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *pre_output = this->Input(PRE_OUTPUT);
    const Tensor *weight = this->Input(WEIGHT);
    const Tensor *bias = this->Input(BIAS);
    const Tensor *pre_cell = this->Input(PRE_CELL);
    Tensor *cell = this->Output(CELL);
    Tensor *output = this->Output(OUTPUT);
    return kernel_->Compute(context, input, pre_output, weight, bias,
                            pre_cell, cell, output);
  }

 private:
  std::unique_ptr<OpenCLLSTMCellKernel> kernel_;

  MACE_OP_INPUT_TAGS(INPUT, PRE_OUTPUT, WEIGHT, BIAS, PRE_CELL);
  MACE_OP_OUTPUT_TAGS(CELL, OUTPUT);
};
#endif

void RegisterLSTMCell(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "LSTMCell", LSTMCellOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "LSTMCell", LSTMCellOp,
                   DeviceType::GPU, half);
}

}  // namespace ops
}  // namespace mace

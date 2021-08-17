// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/qnn/op_builder.h"

#include "mace/core/proto/arg_helper.h"

namespace mace {
class SpaceToDepthOpBuilder : public OpBuilder {
 public:
  explicit SpaceToDepthOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_SPACE_TO_DEPTH);
    SetOpName(op.name().c_str());

    uint32_t block_size = static_cast<uint32_t>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "block_size", 1));
    auto input_shape = graph_builder_->GetTensorShape(op.input(0));
    MACE_CHECK((input_shape[1] % block_size) == 0 &&
               (input_shape[2] % block_size) == 0);
    std::vector<uint32_t> block_size_dims{2};
    std::vector<uint32_t> block_size_data{block_size, block_size};
    AddTensorParam(QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                   block_size_dims, block_size_data.data());

    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterSpaceToDepth(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "SpaceToDepth", SpaceToDepthOpBuilder);
}
}  // namespace qnn
}  // namespace mace

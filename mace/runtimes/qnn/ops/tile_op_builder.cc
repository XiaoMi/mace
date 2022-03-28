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
class TileOpBuilder : public OpBuilder {
 public:
  explicit TileOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_TILE);
    SetOpName(op.name().c_str());

    const Qnn_Tensor_t &input1 = graph_builder_->GetTensor(op.input(1));
    const uint32_t input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    std::vector<uint32_t> dims{input_dims};
    std::vector<uint32_t> multiples(input_dims);
    for (uint32_t i = 0; i < input_dims; ++i) {
      multiples[i] = (reinterpret_cast<int32_t *>(input1.clientBuf.data))[i];
    }
    AddTensorParam(QNN_OP_TILE_PARAM_MULTIPLES, dims, multiples.data());
    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterTile(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Tile", TileOpBuilder);
}
}  // namespace qnn
}  // namespace mace

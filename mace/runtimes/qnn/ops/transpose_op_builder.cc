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
class TransposeOpBuilder : public OpBuilder {
 public:
  explicit TransposeOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_TRANSPOSE);
    SetOpName(op.name().c_str());

    std::vector<int> perm(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "dims"));
    std::vector<uint32_t> perm_data(perm.begin(), perm.end());
    const uint32_t input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    std::vector<uint32_t> dims{input_dims};
    AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM, dims, perm_data.data());

    MACE_CHECK(op.input_size() >= 1);
    AddInput(op.input(0));

    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterTranspose(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Transpose", TransposeOpBuilder);
}
}  // namespace qnn
}  // namespace mace

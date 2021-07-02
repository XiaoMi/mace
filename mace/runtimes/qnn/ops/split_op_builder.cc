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
class SplitOpBuilder : public OpBuilder {
 public:
  explicit SplitOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op) {
    SetOpType(QNN_OP_SPLIT);
    SetOpName(op.name().c_str());

    int axis = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "axis", 3);
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    axis = axis < 0 ? axis + input_dims : axis;
    MACE_CHECK((0 <= axis && axis < input_dims),
               "Expected spliting axis in the range [", -input_dims, ", ",
               input_dims, "], but got ", axis);
    AddScalarParam(
        QNN_OP_SPLIT_PARAM_AXIS,
        {QNN_DATATYPE_UINT_32, .uint32Value = static_cast<uint32_t>(axis)});

    AddInput(op.input(0));
    uint32_t m = op.output_shape_size() - 1;
    std::vector<uint32_t> split_index_dims{m};
    std::vector<uint32_t> split_index(m);
    for (uint32_t i = 0; i < m; ++i) {
      split_index[i] = op.output_shape(i).dims(axis);
    }
    AddTensorParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
                   split_index_dims, split_index.data());
    for (int i = 0; i < op.output_size(); ++i) {
      AddOutput(op.output(i));
    }

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterSplit(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Split", SplitOpBuilder);
}
}  // namespace qnn
}  // namespace mace

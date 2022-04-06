// Copyright 2022 The MACE Authors. All Rights Reserved.
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
class GatherOpBuilder : public OpBuilder {
 public:
  explicit GatherOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_GATHER);
    SetOpName(op.name().c_str());

    MACE_CHECK(op.input_size() == 2 || op.input_size() == 3);
    for (int i = 0; i < 2; ++i) {
      AddInput(op.input(i));
    }
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    if (op.input_size() == 2) {
      int axis = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "axis", 0);
      axis = axis < 0 ? axis + input_dims : axis;
      MACE_CHECK((0 <= axis && axis < input_dims),
                "Expected axis in the range [", -input_dims, ", ",
                input_dims, "], but got ", axis);
      AddScalarParam(
          QNN_OP_GATHER_PARAM_AXIS,
          {QNN_DATATYPE_INT_32, .int32Value = static_cast<int32_t>(axis)});
    } else {
      const Qnn_Tensor_t &axis = graph_builder_->GetTensor(op.input(2));
      int axis_data = (reinterpret_cast<int32_t *>(axis.clientBuf.data))[0];
      axis_data = axis_data < 0 ? axis_data + input_dims : axis_data;
      MACE_CHECK((0 <= axis_data && axis_data < input_dims),
                "Expected axis in the range [", -input_dims, ", ",
                input_dims, "], but got ", axis_data);
      AddScalarParam(
          QNN_OP_GATHER_PARAM_AXIS,
          {QNN_DATATYPE_INT_32, .int32Value = static_cast<int32_t>(axis_data)});
    }
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterGather(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Gather", GatherOpBuilder);
}
}  // namespace qnn
}  // namespace mace

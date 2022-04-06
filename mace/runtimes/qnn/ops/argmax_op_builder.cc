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
class ArgMaxOpBuilder : public OpBuilder {
 public:
  explicit ArgMaxOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    MACE_CHECK(op.input_size() == 2);
    SetOpType(QNN_OP_ARGMAX);
    SetOpName(op.name().c_str());

    const Qnn_Tensor_t &axis = graph_builder_->GetTensor(op.input(1));
    int axis_data = (reinterpret_cast<uint32_t *>(axis.clientBuf.data))[0];
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    const int output_dims = graph_builder_->GetTensorShape(op.output(0)).size();
    MACE_CHECK(output_dims == input_dims - 1);
    if (axis_data < 0) {
      axis_data += input_dims;
    }
    MACE_CHECK((0 <= axis_data && axis_data < input_dims),
               "Expected axis in the range [", -input_dims, ", ",
               input_dims, "], but got ", axis_data);
    AddScalarParam(
        QNN_OP_ARGMAX_PARAM_AXIS,
        {QNN_DATATYPE_UINT_32, .uint32Value = static_cast<uint32_t>(axis_data)});

    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterArgMax(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "ArgMax", ArgMaxOpBuilder);
}
}  // namespace qnn
}  // namespace mace

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
class MatMulOpBuilder : public OpBuilder {
 public:
  explicit MatMulOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_MAT_MUL);
    SetOpName(op.name().c_str());

    bool transpose_a =
        ProtoArgHelper::GetOptionalArg<OperatorDef, bool>(op, "transpose_a", false);
    AddScalarParam(
        QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
        {QNN_DATATYPE_BOOL_8, .bool8Value = static_cast<uint8_t>(transpose_a)});
    bool transpose_b =
        ProtoArgHelper::GetOptionalArg<OperatorDef, bool>(op, "transpose_b", false);
    AddScalarParam(
        QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
        {QNN_DATATYPE_BOOL_8, .bool8Value = static_cast<uint8_t>(transpose_b)});

    MACE_CHECK(op.input_size() == 2);
    const int input0_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    index_t m0 = graph_builder_->GetTensorShape(op.input(0))[input0_dims-2];
    index_t n0 = graph_builder_->GetTensorShape(op.input(0))[input0_dims-1];
    const int input1_dims = graph_builder_->GetTensorShape(op.input(1)).size();
    index_t m1 = graph_builder_->GetTensorShape(op.input(1))[input1_dims-2];
    index_t n1 = graph_builder_->GetTensorShape(op.input(1))[input1_dims-1];
    const int output_dims = graph_builder_->GetTensorShape(op.output(0)).size();
    index_t m2 = graph_builder_->GetTensorShape(op.output(0))[output_dims-2];
    index_t n2 = graph_builder_->GetTensorShape(op.output(0))[output_dims-1];
    if (!transpose_a && !transpose_b) {
      MACE_CHECK(m1 == n0 && m2 == m0 && n2 == n1);
    } else if (!transpose_a && transpose_b) {
      MACE_CHECK(n1 == n0 && m2 == m0 && n2 == m1);
    } else if (transpose_a && !transpose_b) {
      MACE_CHECK(m1 == m0 && m2 == n0 && n2 == n1);
    } else {
      MACE_CHECK(n1 == m0 && m2 == n0 && n2 == m1);
    }

    for (int i = 0; i < op.input_size(); ++i) {
      AddInput(op.input(i));
    }
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterMatMul(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "MatMul", MatMulOpBuilder);
}
}  // namespace qnn
}  // namespace mace

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
#include "mace/core/quantize.h"
#include "mace/ops/common/pad_type.h"

namespace mace {
class PadOpBuilder : public OpBuilder {
 public:
  explicit PadOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_PAD);
    SetOpName(op.name().c_str());

    ops::PadType pad_type = static_cast<ops::PadType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "pad_type", static_cast<int>(ops::PadType::CONSTANT)));
    std::vector<int> paddings(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "paddings"));
    float constant_value = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
        op, "constant_value", 0.0);

    uint32_t pad_scheme = QNN_OP_PAD_SCHEME_CONSTANT;
    if (pad_type == ops::PadType::REFLECT) {
      pad_scheme = QNN_OP_PAD_SCHEME_MIRROR_REFLECT;
    } else if (pad_type == ops::PadType::SYMMETRIC) {
      pad_scheme = QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC;
    }
    AddScalarParam(QNN_OP_PAD_PARAM_SCHEME,
                   {QNN_DATATYPE_UINT_32, .uint32Value = pad_scheme});


    std::vector<uint32_t> paddings_dims{
        static_cast<uint32_t>(paddings.size() / 2), 2};
    std::vector<uint32_t> paddings_data(paddings.begin(), paddings.end());
    AddTensorParam(QNN_OP_PAD_PARAM_PAD_AMOUNT, paddings_dims,
                   paddings_data.data());
    if (pad_type == ops::PadType::CONSTANT) {
      int32_t quantized_constant = Quantize<uint8_t>(
          constant_value, graph_builder_->GetTensorScale(op.input(0)),
          graph_builder_->GetTensorOffset(op.input(0)));
      AddScalarParam(QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
                     {QNN_DATATYPE_INT_32, .int32Value = quantized_constant});
    }

    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterPad(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Pad", PadOpBuilder);
}
}  // namespace qnn
}  // namespace mace

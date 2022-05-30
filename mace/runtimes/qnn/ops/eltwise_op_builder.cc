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
#include "mace/ops/common/eltwise_type.h"

namespace mace {
namespace {
const char *MapEltwiseTypeToQnnOp(ops::EltwiseType type) {
  switch (type) {
    case ops::EltwiseType::SUM:
      return QNN_OP_ELEMENT_WISE_ADD;
    case ops::EltwiseType::SUB:
      return QNN_OP_ELEMENT_WISE_SUBTRACT;
    case ops::EltwiseType::PROD:
      return QNN_OP_ELEMENT_WISE_MULTIPLY;
    case ops::EltwiseType::DIV:
      return QNN_OP_ELEMENT_WISE_DIVIDE;
    case ops::EltwiseType::MIN:
      return QNN_OP_ELEMENT_WISE_MINIMUM;
    case ops::EltwiseType::MAX:
      return QNN_OP_ELEMENT_WISE_MAXIMUM;
    case ops::EltwiseType::NEG:
      return QNN_OP_ELEMENT_WISE_NEG;
    case ops::EltwiseType::ABS:
      return QNN_OP_ELEMENT_WISE_ABS;
    case ops::EltwiseType::SQR_DIFF:
      return QNN_OP_ELEMENT_WISE_SQUARED_DIFFERENCE;
    case ops::EltwiseType::POW:
      return QNN_OP_ELEMENT_WISE_POWER;
    case ops::EltwiseType::EQUAL:
      return QNN_OP_ELEMENT_WISE_EQUAL;
    case ops::EltwiseType::CLIP:
      return QNN_OP_RELU_MIN_MAX;
    default:
      return nullptr;
  }
}
}  // namespace
class EltwiseOpBuilder : public OpBuilder {
 public:
  explicit EltwiseOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    auto type = static_cast<ops::EltwiseType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "type", static_cast<int>(ops::EltwiseType::NONE)));
    VLOG(3) << "EltwiseType: " << static_cast<int>(type);
    const char *op_type = MapEltwiseTypeToQnnOp(type);
    MACE_CHECK_NOTNULL(op_type);
    SetOpType(op_type);
    SetOpName(op.name().c_str());
    if (std::string(op_type) == QNN_OP_RELU_MIN_MAX) {
      float min_value = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
          op, "min", 0);
      float max_value = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
          op, "max", 6);
      AddScalarParam(
        QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
        {QNN_DATATYPE_FLOAT_32, .floatValue = static_cast<float>(min_value)});
      AddScalarParam(
        QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
        {QNN_DATATYPE_FLOAT_32, .floatValue = static_cast<float>(max_value)});
      AddInput(op.input(0));
      AddOutput(op.output(0));
      return MaceStatus::MACE_SUCCESS;
    }
    std::vector<uint32_t> scalar_dims{1};
    if (op.input_size() == 1) {
      float scalar_input = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
          op, "scalar_input", 1.0);
      int scalar_input_index = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          op, "scalar_input_index", 1);
      float scale = 0;
      int32_t zero = 0;
      const std::string scalar_input_name = op.name() + "_scalar";
      if (quantized_type == DT_UINT8) {
        uint8_t quantized_scalar = Quantize<uint8_t>(scalar_input, &scale, &zero);
        graph_builder_->CreateGraphTensor(
            scalar_input_name, 0, QNN_TENSOR_TYPE_STATIC,
            QNN_DATATYPE_UFIXED_POINT_8, scale, zero, scalar_dims,
            &quantized_scalar, 1);
      } else if (quantized_type == DT_UINT16) {
        uint16_t quantized_scalar = Quantize<uint16_t>(scalar_input, &scale, &zero);
        graph_builder_->CreateGraphTensor(
            scalar_input_name, 0, QNN_TENSOR_TYPE_STATIC,
            QNN_DATATYPE_UFIXED_POINT_16, scale, zero, scalar_dims,
            &quantized_scalar, 1);
      } else {
        graph_builder_->CreateGraphTensor(
            scalar_input_name, 0, QNN_TENSOR_TYPE_STATIC,
            QNN_DATATYPE_FLOAT_32, scale, zero, scalar_dims,
            &scalar_input, 1);
      }
      if (scalar_input_index == 0) {
        AddInput(scalar_input_name);
        AddInput(op.input(0));
      } else {
        AddInput(op.input(0));
        if (std::string(op_type) != QNN_OP_ELEMENT_WISE_NEG &&
            std::string(op_type) != QNN_OP_ELEMENT_WISE_ABS) {
          AddInput(scalar_input_name);
        }
      }
    } else {
      AddInput(op.input(0));
      AddInput(op.input(1));
    }

    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterEltwise(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Eltwise", EltwiseOpBuilder);
}
}  // namespace qnn
}  // namespace mace

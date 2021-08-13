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

namespace mace {
const char *MapActivationTypeToQnn(const std::string &type) {
  if (type == "RELU") {
    return QNN_OP_RELU;
  } else if (type == "SIGMOID") {
    return QNN_OP_SIGMOID;
  } else if (type == "TANH") {
    return QNN_OP_TANH;
  } else if (type == "RELUX") {
    return QNN_OP_RELU_MIN_MAX;
  } else if (type == "LEAKYRELU") {
    return QNN_OP_PRELU;
  } else {
    LOG(FATAL) << "Unknown activation type: " << type;
    return nullptr;
  }
}
class ActivationOpBuilder : public OpBuilder {
 public:
  explicit ActivationOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    auto type = ProtoArgHelper::GetOptionalArg<OperatorDef, std::string>(
            op, "activation", "NOOP");
    SetOpType(MapActivationTypeToQnn(type));
    SetOpName(op.name().c_str());
    if (type == "RELUX") {
      AddScalarParam(
        QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
        {QNN_DATATYPE_FLOAT_32, .floatValue = static_cast<float>(0.0)});
      float max_value = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
            op, "max_limit", 6);
      AddScalarParam(
        QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
        {QNN_DATATYPE_FLOAT_32, .floatValue = static_cast<float>(max_value)});
    }
    if (type == "LEAKYRELU") {
      std::vector<uint32_t> coeff_dims{1};
      auto coeff = ProtoArgHelper::GetOptionalArg<OperatorDef, float>(
            op, "activation_coefficient", 0.01);
      float scale = 0;
      int32_t zero = 0;
      const std::string coeff_input_name = op.name() + "_coeff";
      if (quantized_type == DT_UINT8) {
        uint8_t quantized_coeff = Quantize<uint8_t>(coeff, &scale, &zero);
        graph_builder_->CreateGraphTensor(
            coeff_input_name, 0, QNN_TENSOR_TYPE_STATIC,
            QNN_DATATYPE_UFIXED_POINT_8, scale, zero, coeff_dims,
            &quantized_coeff, 1);
      } else {
        uint16_t quantized_coeff = Quantize<uint16_t>(coeff, &scale, &zero);
        graph_builder_->CreateGraphTensor(
            coeff_input_name, 0, QNN_TENSOR_TYPE_STATIC,
            QNN_DATATYPE_UFIXED_POINT_16, scale, zero, coeff_dims,
            &quantized_coeff, 1);
      }
      AddInput(op.input(0));
      AddInput(coeff_input_name);
    } else {
      AddInput(op.input(0));
    }
    AddOutput(op.output(0));
    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterActivation(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Activation", ActivationOpBuilder);
}
}  // namespace qnn
}  // namespace mace

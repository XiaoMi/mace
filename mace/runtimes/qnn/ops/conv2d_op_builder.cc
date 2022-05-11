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
#include "mace/runtimes/qnn/ops/conv_pool_2d_util.h"

namespace mace {
class Conv2dOpBuilder : public OpBuilder {
 public:
  explicit Conv2dOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {
    names_ = {
        {"Conv2D",
         {QNN_OP_CONV_2D, QNN_OP_CONV_2D_PARAM_STRIDE,
          QNN_OP_CONV_2D_PARAM_DILATION, QNN_OP_CONV_2D_PARAM_PAD_AMOUNT}},
        {"DepthwiseConv2d",
         {QNN_OP_DEPTH_WISE_CONV_2D, QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE,
          QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION,
          QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT}}};
  }

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    MACE_CHECK(names_.count(op.type()) > 0,
               "QNN does not support op: ", op.type());
    auto names = names_.at(op.type());
    SetOpType(names.op_type);
    SetOpName(op.name().c_str());

    std::vector<int> strides(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "strides"));
    MACE_CHECK(strides.size() == 2);
    std::vector<uint32_t> strides_dims{2};
    std::vector<uint32_t> strides_data(strides.begin(), strides.end());
    AddTensorParam(names.stride, strides_dims, strides_data.data());

    std::vector<int> dilations(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "dilations",
                                                          {1, 1}));
    std::vector<uint32_t> dilations_dims{2};
    std::vector<uint32_t> dilations_data(dilations.begin(), dilations.end());
    AddTensorParam(names.dilation, dilations_dims, dilations_data.data());

    std::vector<int> paddings(ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(
        op, "padding_values"));
    std::vector<uint32_t> paddings_data;
    CalcPadding(graph_builder_->GetTensorShape(op.input(0)).data(),
                graph_builder_->GetTensorShape(op.input(1)).data(),
                graph_builder_->GetTensorShape(op.output(0)).data(),
                dilations.data(), strides.data(), paddings, &paddings_data);
    std::vector<uint32_t> paddings_dims{2, 2};
    AddTensorParam(names.pad_amount, paddings_dims, paddings_data.data());

    const std::string act_type =
        ProtoArgHelper::GetOptionalArg<OperatorDef, std::string>(
            op, "activation", "NOOP");
    MACE_CHECK(act_type == "NOOP" || act_type == "RELU" || act_type == "RELUX");

    AddInput(op.input(0));
    AddInput(op.input(1));
    index_t output_channels = graph_builder_->GetTensorShape(op.output(0))[3];
    std::vector<uint32_t> bias_dims{static_cast<uint32_t>(output_channels)};
    std::vector<int32_t> bias_data(output_channels, 0);
    if (op.input_size() == 3) {
      AddInput(op.input(2));
    } else {
      const float bias_scale = graph_builder_->GetTensorScale(op.input(0)) *
                               graph_builder_->GetTensorScale(op.input(1));
      const std::string bias_name = op.name() + "_bias";
      auto data_type = static_cast<DataType>(ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
              op, "T", static_cast<int>(DT_UINT8)));
      if (data_type == DT_FLOAT) {
        graph_builder_->CreateGraphTensor(
            bias_name, 0, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
            bias_scale, 0, bias_dims, bias_data.data(),
            static_cast<uint32_t>(bias_data.size() * sizeof(bias_data[0])));
      } else {
        graph_builder_->CreateGraphTensor(
            bias_name, 0, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_32,
            bias_scale, 0, bias_dims, bias_data.data(),
            static_cast<uint32_t>(bias_data.size() * sizeof(bias_data[0])));
      }
      AddInput(bias_name);
    }
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  struct Names {
    const char *op_type;
    const char *stride;
    const char *dilation;
    const char *pad_amount;
  };
  std::unordered_map<std::string, Names> names_;
};
namespace qnn {
void RegisterConv2D(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Conv2D", Conv2dOpBuilder);
  QNN_REGISTER_OP(op_registry, "DepthwiseConv2d", Conv2dOpBuilder);
}
}  // namespace qnn
}  // namespace mace

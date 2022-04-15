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
#include "mace/ops/common/reduce_type.h"

namespace mace {
class ReduceOpBuilder : public OpBuilder {
 public:
  explicit ReduceOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {
    names_ = {
        {ReduceType::MEAN,
         {QNN_OP_REDUCE_MEAN, QNN_OP_REDUCE_MEAN_PARAM_AXES,
          QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS}},
        {ReduceType::MIN,
         {QNN_OP_REDUCE_MIN, QNN_OP_REDUCE_MIN_PARAM_AXES,
          QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS}},
        {ReduceType::MAX,
         {QNN_OP_REDUCE_MAX, QNN_OP_REDUCE_MAX_PARAM_AXES,
          QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS}},
        {ReduceType::PROD,
         {QNN_OP_REDUCE_PROD, QNN_OP_REDUCE_PROD_PARAM_AXES,
          QNN_OP_REDUCE_PROD_PARAM_KEEP_DIMS}},
        {ReduceType::SUM,
         {QNN_OP_REDUCE_SUM, QNN_OP_REDUCE_SUM_PARAM_AXES,
          QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS}},
    };
  }

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    auto type = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op, "reduce_type", static_cast<int>(ReduceType::MEAN));
    MACE_CHECK(names_.count(type) > 0, "QNN does not support reduce: ", type);
    auto names = names_.at(type);
    SetOpType(names.op_type);
    SetOpName(op.name().c_str());

    std::vector<int> axes(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "axis"));
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    MACE_CHECK((0 < axes.size() && static_cast<int>(axes.size()) <= input_dims),
               "Expected axis num in the range [", 1, ", ", input_dims,
               "], but got ", axes.size());
    for (auto &axis : axes) {
      axis = axis < 0 ? axis + input_dims : axis;
      MACE_CHECK((0 <= axis && axis < input_dims),
                 "Expected reducing axis in the range [", -input_dims, ", ",
                 input_dims, "], but got ", axis);
    }
    std::vector<uint32_t> axes_dims{static_cast<uint32_t>(axes.size())};
    std::vector<uint32_t> axes_data(axes.begin(), axes.end());
    AddTensorParam(names.axes, axes_dims, axes_data.data());

    int keepdims =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "keepdims", 0);
    AddScalarParam(
        names.keepdims,
        {QNN_DATATYPE_BOOL_8, .bool8Value = static_cast<uint8_t>(keepdims)});

    MACE_CHECK(op.input_size() == 1);
    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  struct Names {
    const char *op_type;
    const char *axes;
    const char *keepdims;
  };
  std::unordered_map<int, Names> names_;
};
namespace qnn {
void RegisterReduce(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Reduce", ReduceOpBuilder);
}
}  // namespace qnn
}  // namespace mace

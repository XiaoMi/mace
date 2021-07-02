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
class MomentsOpBuilder : public OpBuilder {
 public:
  explicit MomentsOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op) {
    SetOpType(QNN_OP_MOMENTS);
    SetOpName(op.name().c_str());

    std::vector<int> axes(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "axis"));
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    MACE_CHECK((0 < axes.size() && static_cast<int>(axes.size()) <= input_dims),
               "Expected axis num in the range [", 1, ", ", input_dims,
               "], but got ", axes.size());
    for (auto axis : axes) {
      axis = axis < 0 ? axis + input_dims : axis;
      MACE_CHECK((0 <= axis && axis < input_dims),
                 "Expected reducing axis in the range [", -input_dims, ", ",
                 input_dims, "], but got ", axis);
    }
    std::vector<uint32_t> axes_dims{static_cast<uint32_t>(axes.size())};
    std::vector<uint32_t> axes_data(axes.begin(), axes.end());
    AddTensorParam(QNN_OP_MOMENTS_PARAM_AXES, axes_dims, axes_data.data(),
                   QNN_DATATYPE_INT_32);

    int keepdims =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "keepdims", 0);
    AddScalarParam(
        QNN_OP_MOMENTS_PARAM_KEEP_DIMS,
        {QNN_DATATYPE_BOOL_8, .bool8Value = static_cast<uint8_t>(keepdims)});

    MACE_CHECK(op.input_size() == 1);
    AddInput(op.input(0));
    AddOutput(op.output(0));
    AddOutput(op.output(1));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterMoments(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Moments", MomentsOpBuilder);
}
}  // namespace qnn
}  // namespace mace

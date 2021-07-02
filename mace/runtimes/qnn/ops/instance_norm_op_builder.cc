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

class InstanceNormOpBuilder : public OpBuilder {
 public:
  explicit InstanceNormOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {
    names_ = {
        {"InstanceNorm",
         {QNN_OP_INSTANCE_NORM, QNN_OP_INSTANCE_NORM_PARAM_EPSILON,
          QNN_OP_INSTANCE_NORM_PARAM_MODE, QNN_OP_INSTANCE_NORM_PARAM_REGION}}};
  }

  MaceStatus BuildOp(const OperatorDef &op) {
    MACE_CHECK(names_.count(op.type()) > 0,
               "QNN does not support op: ", op.type());
    auto names = names_.at(op.type());
    SetOpType(names.op_type);
    SetOpName(op.name().c_str());

    float epsilon =
      ProtoArgHelper::GetOptionalArg<OperatorDef, float>(op, "epsilon", 0);
    AddScalarParam(names.epsilon,
                   {QNN_DATATYPE_FLOAT_32,
                    .floatValue = static_cast<float>(epsilon)});
    AddInput(op.input(0));
    AddInput(op.input(1));
    AddInput(op.input(2));
    AddOutput(op.output(0));
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  struct Names {
    const char *op_type;
    const char *epsilon;
    const char *mode;
    const char *region;
  };
  std::unordered_map<std::string, Names> names_;
};
namespace qnn {
void RegisterInstanceNorm(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "InstanceNorm", InstanceNormOpBuilder);
}
}  // namespace qnn
}  // namespace mace

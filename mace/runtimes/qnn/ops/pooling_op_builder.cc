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
#include "mace/ops/common/pooling_type.h"

namespace mace {
class PoolingOpBuilder : public OpBuilder {
 public:
  explicit PoolingOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {
    names_ = {{PoolingType::AVG,
               {QNN_OP_POOL_AVG_2D, QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
                QNN_OP_POOL_AVG_2D_PARAM_STRIDE,
                QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT}},
              {PoolingType::MAX,
               {QNN_OP_POOL_MAX_2D, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
                QNN_OP_POOL_MAX_2D_PARAM_STRIDE,
                QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT}}};
  }

  MaceStatus BuildOp(const OperatorDef &op) {
    auto pooling_type =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "pooling_type", static_cast<int>(PoolingType::AVG));
    MACE_CHECK(names_.count(pooling_type) > 0,
               "QNN does not support pooling type: ", pooling_type);
    auto names = names_.at(pooling_type);
    SetOpType(names.op_type);
    SetOpName(op.name().c_str());

    std::vector<int> kernels(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "kernels"));
    std::vector<uint32_t> kernels_dims{2};
    std::vector<uint32_t> kernels_data(kernels.begin(), kernels.end());
    AddTensorParam(names.filter_size, kernels_dims, kernels_data.data());

    std::vector<int> strides(
        ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(op, "strides"));
    MACE_CHECK(strides.size() == 2);
    std::vector<uint32_t> strides_dims{2};
    std::vector<uint32_t> strides_data(strides.begin(), strides.end());
    AddTensorParam(names.stride, strides_dims, strides_data.data());

    std::vector<int> dilations{1, 1};
    std::vector<int> paddings(ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(
        op, "padding_values"));
    std::vector<uint32_t> paddings_data;
    auto input_shape = graph_builder_->GetTensorShape(op.input(0));
    std::vector<uint32_t> filter_shape = {kernels_data[0], kernels_data[1],
                                          input_shape[3], input_shape[3]};
    CalcPadding(input_shape.data(), filter_shape.data(),
                graph_builder_->GetTensorShape(op.output(0)).data(),
                dilations.data(), strides.data(), paddings, &paddings_data);
    std::vector<uint32_t> paddings_dims{2, 2};
    AddTensorParam(names.pad_amount, paddings_dims, paddings_data.data());

    AddInput(op.input(0));
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  struct Names {
    const char *op_type;
    const char *filter_size;
    const char *stride;
    const char *pad_amount;
  };
  std::unordered_map<int, Names> names_;
};
namespace qnn {
void RegisterPooling(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Pooling", PoolingOpBuilder);
}
}  // namespace qnn
}  // namespace mace

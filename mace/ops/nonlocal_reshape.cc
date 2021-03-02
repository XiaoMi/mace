// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer/reshape.h"
#include "mace/ops/opencl/image/reshape.h"
#endif

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class NonlocalReshapeOp : public Operation {
 public:
  explicit NonlocalReshapeOp(OpConstructContext *context)
      : Operation(context),
        has_df_(Operation::GetOptionalArg<int>("has_data_format", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *shape = this->Input(SHAPE);
    std::vector<index_t> out_shape;
    for (int i = 0; i < 2; i++) {
      out_shape.push_back(input->dim(i));
    }
    for (int i = 2; i < 4; i++) {
      out_shape.push_back(shape->dim(i));
    }

    Tensor *output = this->Output(OUTPUT);
    output->ReuseTensorBuffer(*input);
    output->Reshape(out_shape);

    return MaceStatus::MACE_SUCCESS;
  }

  int ReuseTensorMapId(size_t output_idx) const override {
    if (output_idx == 0) {
      return 0;
    } else {
      return Operation::ReuseTensorMapId(output_idx);
    }
  }

 private:
  bool has_df_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template <>
class NonlocalReshapeOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit NonlocalReshapeOp(OpConstructContext *context)
      : Operation(context), mem_type_(context->GetOpMemoryType()) {
    if (mem_type_ == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ReshapeKernel>(context);
    } else {
      kernel_ = make_unique<opencl::buffer::ReshapeKernel>();
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *shape = this->Input(SHAPE);
    std::vector<index_t> out_shape;
    for (int i = 0; i < 2; i++) {
      out_shape.push_back(input->dim(i));
    }
    for (int i = 2; i < 4; i++) {
      out_shape.push_back(shape->dim(i));
    }
    Tensor *output = this->Output(OUTPUT);
    return kernel_->Compute(context, input, out_shape, output);
  }

  int ReuseTensorMapId(size_t output_idx) const override {
    if (output_idx == 0 && mem_type_ == MemoryType::GPU_BUFFER) {
      return 0;
    } else {
      return Operation::ReuseTensorMapId(output_idx);
    }
  }

 private:
  std::unique_ptr<OpenCLReshapeKernel> kernel_;
  MemoryType mem_type_;
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif

void RegisterNonlocalReshape(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "NonlocalReshape",
                   NonlocalReshapeOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_OP(op_registry, "NonlocalReshape",
                   NonlocalReshapeOp, RuntimeType::RT_CPU, int32_t);
  MACE_REGISTER_GPU_OP(op_registry, "NonlocalReshape", NonlocalReshapeOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry, OpConditionBuilder("NonlocalReshape").SetDevicePlacerFunc(
      [](OpConditionContext *context) -> std::set<RuntimeType> {
        auto op = context->operator_def();
        if (op->output_shape_size() != op->output_size()) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        // When transforming a model, has_data_format is set
        // to true only when the data dimension conforms to
        // specific rules, such as dimension == 4
        int has_data_format =
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *op, "has_data_format", 0);
        if (has_data_format) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        DataFormat op_data_format = static_cast<DataFormat>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *context->operator_def(), "data_format",
                static_cast<int>(DataFormat::NONE)));
        auto tensor_shape_info = context->tensor_shape_info();
        const std::string &input_0 = op->input(0);
        const auto out_dims_size =
            op->output_shape(0).dims_size();
        if (op_data_format == DataFormat::NHWC &&
            4 == tensor_shape_info->at(input_0).size() &&
            (out_dims_size == 4 || out_dims_size == 2)) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        return {RuntimeType::RT_CPU};
      }));
}

}  // namespace ops
}  // namespace mace

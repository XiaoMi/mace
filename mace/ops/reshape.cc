// Copyright 2018 The MACE Authors. All Rights Reserved.
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

namespace {

MaceStatus GetOutputShape(const Tensor *input,
                          const int32_t *shape_data,
                          const index_t num_dims,
                          std::vector<index_t> *out_shape) {
  MACE_CHECK(input != nullptr && shape_data != nullptr && out_shape != nullptr);
  int unknown_idx = -1;
  index_t product = 1;
  index_t n = 0;

  out_shape->resize(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    if (shape_data[i] == -1) {
      MACE_CHECK(unknown_idx == -1, "Only one input size may be -1");
      unknown_idx = i;
      (*out_shape)[i] = 1;
    } else {
      MACE_CHECK(shape_data[i] >= 0, "Shape must be non-negative: ",
                 shape_data[i]);
      if (shape_data[i] == 0) {
        MACE_CHECK(i < input->dim_size(), "dims:0 out of input dims' range.");
        n = input->dim(i);
      } else {
        n = shape_data[i];
      }
      (*out_shape)[i] = n;
      product *= n;
    }
  }

  if (unknown_idx != -1) {
    MACE_CHECK(product != 0)
      << "Cannot infer shape if there is zero shape size.";
    const index_t missing = input->size() / product;
    MACE_CHECK(missing * product == input->size())
      << "Input size not match reshaped tensor size";
    (*out_shape)[unknown_idx] = missing;
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace

template<RuntimeType D, class T>
class ReshapeOp : public Operation {
 public:
  explicit ReshapeOp(OpConstructContext *context)
      : Operation(context), dim_(Operation::GetRepeatedArgs<int>("dim")),
        has_df_(Operation::GetOptionalArg<int>("has_data_format", 0)) {}

  // We reuse tensors, so not need to Map input/output.
  MaceStatus Forward(OpContext *context) override {
    if (dim_.size() == 0 && InputSize() > 1) {
      const Tensor *shape = this->Input(SHAPE);
      Tensor::MappingGuard shape_guard(shape);
      return Run(context);
    } else {
      // Use dim_, not need to map shape tensor.
      return Run(context);
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    std::vector<int32_t> trans_shape_data;
    if (dim_.size() == 0) {
      const Tensor *shape = this->Input(SHAPE);
      auto shape_data = shape->data<int32_t>();
      trans_shape_data.assign(shape_data, shape_data + shape->size());
    } else {
      trans_shape_data = dim_;
    }

    // NHWC -> NCHW
    if (has_df_ && D == RuntimeType::RT_CPU && trans_shape_data.size() == 4) {
      std::vector<int> dst_dims = {0, 3, 1, 2};
      std::vector<int32_t> tmp_shape =
          TransposeShape<int32_t, int32_t>(trans_shape_data, dst_dims);
      trans_shape_data = tmp_shape;
    }

    std::vector<index_t> out_shape;
    MACE_RETURN_IF_ERROR(GetOutputShape(input, trans_shape_data.data(),
                                        trans_shape_data.size(), &out_shape));

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
  std::vector<int> dim_;
  bool has_df_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template <>
class ReshapeOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit ReshapeOp(OpConstructContext *context)
      : Operation(context), dim_(Operation::GetRepeatedArgs<int>("dim")),
        mem_type_(context->GetOpMemoryType()) {
    if (mem_type_ == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ReshapeKernel>(context);
    } else {
      kernel_ = make_unique<opencl::buffer::ReshapeKernel>();
    }
  }

  int ReuseTensorMapId(size_t output_idx) const override {
    if (output_idx == 0 && mem_type_ == MemoryType::GPU_BUFFER) {
      return 0;
    } else {
      return Operation::ReuseTensorMapId(output_idx);
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const int32_t *shape_data = dim_.data();
    const index_t num_dims = dim_.size();
    std::vector<index_t> out_shape;
    MACE_RETURN_IF_ERROR(
        GetOutputShape(input, shape_data, num_dims, &out_shape));

    Tensor *output = this->Output(OUTPUT);
    return kernel_->Compute(context, input, out_shape, output);
  }

 private:
  std::vector<int> dim_;
  std::unique_ptr<OpenCLReshapeKernel> kernel_;
  MemoryType mem_type_;
  MACE_OP_INPUT_TAGS(INPUT, SHAPE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif

void RegisterReshape(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Reshape", ReshapeOp, RuntimeType::RT_CPU);
  MACE_REGISTER_OP(op_registry, "Reshape", ReshapeOp,
                   RuntimeType::RT_CPU, int32_t);
  MACE_REGISTER_GPU_OP(op_registry, "Reshape", ReshapeOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry, OpConditionBuilder("Reshape").SetDevicePlacerFunc(
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
        if (has_data_format && op->input_size() == 1) {
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
            (out_dims_size == 4 || out_dims_size == 2) &&
            op->input_size() == 1) {
          return {RuntimeType::RT_CPU, RuntimeType::RT_OPENCL};
        }

        return {RuntimeType::RT_CPU};
      }));
}

}  // namespace ops
}  // namespace mace

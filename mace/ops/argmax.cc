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

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class ArgMaxOp : public Operation {
 public:
  explicit ArgMaxOp(OpConstructContext *context)
      : Operation(context),
        model_type_(static_cast<FrameworkType>(Operation::GetOptionalArg<int>(
            "framework_type", FrameworkType::TENSORFLOW))),
        has_axis_(model_type_ != FrameworkType::CAFFE ||
            Operation::ExistArg("axis")),
        top_k_(Operation::GetOptionalArg<int>("top_k", 1)),
        out_val_(Operation::GetOptionalArg<bool>("out_val", false)),
        axis_(Operation::GetOptionalArg<int>("axis", 0)),
        argmin_(Operation::GetOptionalArg<bool>("argmin", false)),
        keep_dims_(Operation::GetOptionalArg<bool>("keepdims", true)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    const auto input_dim_size = input->dim_size();
    MACE_CHECK(input_dim_size > 0, "ArgMax input should not be a scalar");
    const auto axis_value = GetAxisValue(input_dim_size);
    MACE_RETURN_IF_ERROR(ResizeOutputTensor(output, input, axis_value));

    auto input_data = input->data<T>();

    int axis_dim = 0;
    int axis_dist = 0;
    const auto &input_shape = input->shape();
    if (axis_value != 0) {
      axis_dim = input->dim(axis_value);
      axis_dist = std::accumulate(input_shape.begin() + axis_value,
                                  input_shape.end(),
                                  1, std::multiplies<int>()) / axis_dim;
    } else {
      axis_dim = input->dim(0);
      axis_dist = 1;
    }
    const auto output_loop = input->size() / axis_dim;

    for (int i = 0; i < output_loop; i += 1) {
      std::vector<std::pair<T, int>> input_data_vector(axis_dim);
      const auto axis_base = i / axis_dist * axis_dim;
      const auto axis_offset = i % axis_dist;
      for (int d = 0; d < axis_dim; ++d) {
        const auto input_idx = (axis_base + d) * axis_dist + axis_offset;
        input_data_vector[d] = std::make_pair(input_data[input_idx], d);
      }

      if (argmin_) {
        std::partial_sort(input_data_vector.begin(),
                          input_data_vector.begin() + top_k_,
                          input_data_vector.end(),
                          std::less<std::pair<T, int>>());
      } else {
        std::partial_sort(input_data_vector.begin(),
                          input_data_vector.begin() + top_k_,
                          input_data_vector.end(),
                          std::greater<std::pair<T, int>>());
      }

      if (!out_val_) {
        auto output_data = output->mutable_data<int32_t>();
        const auto top_k_base = i / axis_dist * top_k_;
        for (int j = 0; j < top_k_; ++j) {
          const auto output_idx = (top_k_base + j) * axis_dist + axis_offset;
          output_data[output_idx] = input_data_vector[j].second;
        }
      } else if (has_axis_) {  // Produces max/min value per axis
        auto output_data = output->mutable_data<T>();
        const auto top_k_base = i / axis_dist * top_k_;
        for (int j = 0; j < top_k_; ++j) {
          auto output_idx = (top_k_base + j) * axis_dist + axis_offset;
          output_data[output_idx] = input_data_vector[j].first;
        }
      } else {  // Produces max_ind and max/min value
        auto output_data = output->mutable_data<T>();
        const auto top_k_base_pos = 2 * i * top_k_;
        const auto top_k_base_value = top_k_base_pos + top_k_;
        for (int j = 0; j < top_k_; ++j) {
          output_data[top_k_base_pos + j] = input_data_vector[j].second;
          output_data[top_k_base_value + j] = input_data_vector[j].first;
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int GetAxisValue(const index_t input_dim_size) {
    const Tensor *axis = this->InputSize() == 2 ? this->Input(1) : nullptr;
    int axis_value = 0;
    if (axis != nullptr) {
      MACE_CHECK(axis->dim_size() == 0,
                 "Mace argmax only supports scalar axis");
      axis_value = axis->data<int32_t>()[0];
    } else {
      axis_value = axis_;
    }
    if (axis_value < 0) {
      axis_value += input_dim_size;
    }

    return axis_value;
  }

  MaceStatus ResizeOutputTensor(Tensor *output, const Tensor *input,
                                const index_t axis_value) {
    auto &input_shape = input->shape();
    std::vector<index_t> output_shape;
    if (model_type_ == FrameworkType::CAFFE) {
      auto output_dim_num = input_shape.size();
      if (output_dim_num < 3) {
        output_dim_num = 3;
      }
      output_shape.assign(output_dim_num, 1);
      if (has_axis_) {
        // Produces max/min idx or max/min value per axis
        output_shape.assign(input_shape.begin(), input_shape.end());
        output_shape[axis_value] = top_k_;
      } else {
        output_shape[0] = input_shape[0];
        // Produces max_ind
        output_shape[2] = top_k_;
        if (out_val_) {
          // Produces max/min idx and max/min value
          output_shape[1] = 2;
        }
      }
    } else {  // for Tensorflow and ONNX
      output_shape.assign(input_shape.begin(),
                          input_shape.begin() + axis_value);
      if (keep_dims_) {
        output_shape.push_back(1);
      }
      for (size_t d = axis_value + 1; d < input_shape.size(); ++d) {
        output_shape.push_back(input_shape[d]);
      }
    }

    return output->Resize(output_shape);
  }

 protected:
  const FrameworkType model_type_;
  // for Caffe
  const bool has_axis_;
  const int top_k_;
  const bool out_val_;

  // for ONNX and TENSORFLOW
  const int axis_;
  const bool argmin_;

  // for ONNX
  const bool keep_dims_;
};

void RegisterArgMax(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ArgMax", ArgMaxOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ArgMax", ArgMaxOp, RuntimeType::RT_CPU);
}

}  // namespace ops
}  // namespace mace

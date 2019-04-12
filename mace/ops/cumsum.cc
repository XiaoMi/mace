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

#include <functional>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class CumsumOp;

template <typename T>
class CumsumOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit CumsumOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 0)),
        exclusive_(Operation::GetOptionalArg<bool>("exclusive", false)),
        reverse_(Operation::GetOptionalArg<bool>("reverse", false)),
        checked_(false) {}

  void Validate() {
    const int32_t input_dims = this->Input(0)->dim_size();
    axis_ =
        axis_ < 0 ? axis_ + input_dims : axis_;
    MACE_CHECK((0 <= axis_ && axis_ < input_dims),
               "Expected concatenating axis in the range [", -input_dims, ", ",
               input_dims, "], but got ", axis_);
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    if (!checked_) {
      Validate();
      bool has_data_format = Operation::GetOptionalArg<int>(
          "has_data_format", 0);
      if (has_data_format && this->Input(0)->dim_size() == 4) {
        if (axis_ == 3) axis_ = 1;
        else if (axis_ == 2) axis_ = 3;
        else if (axis_ == 1) axis_ = 2;
      }
      checked_ = true;
    }

    const Tensor *input = this->Input(0);
    const std::vector<index_t> input_shape = input->shape();

    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    float *output_ptr = output->mutable_data<float>();

    const index_t outer_size = std::accumulate(input_shape.begin(),
                                               input_shape.begin() + axis_,
                                               1,
                                               std::multiplies<index_t>());
    const index_t inner_size = std::accumulate(input_shape.begin() + axis_ + 1,
                                               input_shape.end(),
                                               1,
                                               std::multiplies<index_t>());
    const index_t cum_size = input_shape[axis_];

    if (!reverse_) {
      for (index_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        index_t start_idx = outer_idx * cum_size * inner_size;
        for (index_t cum_idx = 0; cum_idx < cum_size; ++cum_idx) {
          if (cum_idx == 0) {
            if (exclusive_) {
              std::memset(output_ptr + start_idx,
                          0,
                          sizeof(T) * inner_size);
            } else {
              std::memcpy(output_ptr + start_idx,
                          input_ptr + start_idx,
                          sizeof(T) * inner_size);
            }
          } else {
            index_t cur_idx = start_idx + cum_idx * inner_size;
            index_t pre_idx = start_idx + (cum_idx - 1) * inner_size;
            index_t input_idx = exclusive_ ? pre_idx : cur_idx;
            for (index_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
              output_ptr[cur_idx + inner_idx] =
                output_ptr[pre_idx + inner_idx] +
                input_ptr[input_idx + inner_idx];
            }
          }
        }
      }
    } else {
      for (index_t outer_idx = outer_size - 1; outer_idx >= 0; --outer_idx) {
        index_t start_idx = outer_idx * cum_size * inner_size;
        for (index_t cum_idx = cum_size - 1; cum_idx >= 0; --cum_idx) {
          index_t cur_idx = start_idx + cum_idx * inner_size;
          if (cum_idx == cum_size - 1) {
            if (exclusive_) {
              std::memset(output_ptr + cur_idx,
                          0,
                          sizeof(T) * inner_size);
            } else {
              std::memcpy(output_ptr + cur_idx,
                          input_ptr + cur_idx,
                          sizeof(T) * inner_size);
            }
          } else {
            index_t pre_idx = start_idx + (cum_idx + 1) * inner_size;
            index_t input_idx = exclusive_ ? pre_idx : cur_idx;
            for (index_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
              output_ptr[cur_idx + inner_idx] =
                output_ptr[pre_idx + inner_idx] +
                input_ptr[input_idx + inner_idx];
            }
          }
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int32_t axis_;
  bool exclusive_;
  bool reverse_;
  bool checked_;
};

void RegisterCumsum(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Cumsum", CumsumOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace

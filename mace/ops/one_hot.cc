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
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

class OneHotOpBase : public Operation {
 public:
  explicit OneHotOpBase(OpConstructContext *context)
      : Operation(context),
        depth_(Operation::GetOptionalArg<int>("depth", 0)),
        on_value_(Operation::GetOptionalArg<float>("on_value", 1)),
        off_value_(Operation::GetOptionalArg<float>("off_value", 0)),
        axis_(Operation::GetOptionalArg<int>("axis", -1)) {
    MACE_CHECK(depth_ > 0);
  }

 protected:
  int depth_;
  float on_value_;
  float off_value_;
  int axis_;
};

template <DeviceType D, typename T>
class OneHotOp;

template <typename T>
class OneHotOp<DeviceType::CPU, T> : public OneHotOpBase {
 public:
  explicit OneHotOp(OpConstructContext *context) : OneHotOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    index_t axis = axis_ == -1 ? input->dim_size() : axis_;
    const std::vector<index_t> &input_shape = input->shape();
    std::vector<index_t> output_shape(input_shape.size() + 1);

    MACE_CHECK(input->dim_size() < 100);  // prevents too deep recursion
    MACE_CHECK(axis >= 0 && axis <= input->dim_size());

    for (size_t in = 0, out = 0; out < output_shape.size(); ++out) {
      if (static_cast<index_t>(out) == axis) {
        output_shape[out] = depth_;

      } else {
        output_shape[out] = input_shape[in];
        ++in;
      }
    }

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    if (input_shape.size() == 1) {
      const index_t batch = input->dim(0);

      if (axis == 1) {
        for (index_t i = 0; i < batch; ++i) {
          for (index_t j = 0; j < depth_; ++j) {
            output_ptr[i * depth_ + j] = input_ptr[i] == j ? on_value_ :
                                                             off_value_;
          }
        }
      } else {
        for (index_t i = 0; i < depth_; ++i) {
          for (index_t j = 0; j < batch; ++j) {
            output_ptr[i * batch + j] = input_ptr[j] == i ? on_value_ :
                                                            off_value_;
          }
        }
      }
    } else {
      run(input, &input_ptr, &output_ptr, axis, 0, 0, input_shape.size(), 0);
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void run(const Tensor *input, const T **input_ptr,
           T **output_ptr, const index_t axis,
           const index_t current_in, const index_t current_out,
           const index_t left, const index_t test) const {
    if (current_out == axis) {
      const index_t length = depth_;

      if (left == 0) {
        for (index_t i = 0; i < length; ++i) {
          **output_ptr = **input_ptr == i ? on_value_ : off_value_;
          ++(*output_ptr);
        }

        ++(*input_ptr);

      } else {
        const T *in = *input_ptr;

        for (index_t i = 0; i < length; ++i) {
          *input_ptr = in;
          run(input, input_ptr, output_ptr, axis, current_in,
              current_out + 1, left - 1, i);
        }
      }
    } else {
      const index_t length = input->dim(current_in);

      if (left == 0) {
        for (index_t i = 0; i < length; ++i) {
          **output_ptr = **input_ptr == test ? on_value_ : off_value_;
          ++(*output_ptr);
          ++(*input_ptr);
        }
      } else {
        for (index_t i = 0; i < length; ++i) {
          run(input, input_ptr, output_ptr, axis, current_in + 1,
              current_out + 1, left - 1, test);
        }
      }
    }
  }
};


void RegisterOneHot(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "OneHot", OneHotOp, DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace

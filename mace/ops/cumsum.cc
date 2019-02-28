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

namespace {
void PlusOne(int* val) {
  ++(*val);
}

void SubOne(int* val) {
  --(*val);
}

bool LessThan(const int& val, const int& boundary) {
  return val < boundary;
}

bool NotLessThanZero(const int& val, const int& boundary) {
  MACE_UNUSED(boundary);
  return val >= 0;
}

}  // namespace

template <DeviceType D, typename T>
class CumsumOp;

template <typename T>
class CumsumOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit CumsumOp(OpConstructContext *context)
      : Operation(context),
        axis_(Operation::GetOptionalArg<int>("axis", 3)),
        exclusive_(Operation::GetOptionalArg<bool>("exclusive", false)),
        reverse_(Operation::GetOptionalArg<bool>("reverse", false)) {}

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
      auto df = static_cast<DataFormat>(Operation::GetOptionalArg<int>(
          "data_format", DataFormat::DF_NONE));
      if (df == DataFormat::NHWC && this->Input(0)->dim_size() == 4) {
        if (axis_ == 3) axis_ = 1;
        else if (axis_ == 2) axis_ = 3;
        else if (axis_ == 1) axis_ = 2;
      }
      checked_ = true;
    }

    const Tensor *input = this->Input(0);

    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    float *output_ptr = output->mutable_data<float>();

    std::function<void(int*)> next = reverse_ ? SubOne : PlusOne;
    std::function<void(int*)> previous = reverse_ ? PlusOne : SubOne;
    std::function<bool(const int&, const int&)> boundary =
      reverse_ ? NotLessThanZero : LessThan;

    if (input->dim_size() == 4) {
      const int batch = input->dim(0);
      const int channel = input->dim(1);
      const int height = input->dim(2);
      const int width = input->dim(3);

      const int axis_dim_size = input->dim(axis_);

      for (int n = reverse_ ? batch - 1 : 0; boundary(n, batch); next(&n)) {
        for (int c = reverse_ ? channel - 1 : 0; boundary(c, channel);
             next(&c)) {
          for (int h = reverse_ ? height - 1 : 0; boundary(h, height);
               next(&h)) {
            for (int w = reverse_ ? width - 1 : 0; boundary(w, width);
                 next(&w)) {
              int dims[4] = {n, c, h, w};
              if (!reverse_ && dims[axis_] == 0) {
                if (exclusive_) {
                  output_ptr[((n * channel + c) * height + h) * width + w] = 0;
                } else {
                  continue;
                }
              } else if (reverse_ && dims[axis_] == axis_dim_size - 1) {
                if (exclusive_) {
                  output_ptr[((n * channel + c) * height + h) * width + w] = 0;
                } else {
                  continue;
                }
              } else {
                previous(&dims[axis_]);
                if (exclusive_) {
                  output_ptr[((n * channel + c) * height + h) * width + w] =
                      input_ptr[((dims[0] * channel + dims[1]) * height +
                                 dims[2]) *
                                    width +
                                dims[3]] +
                      output_ptr[((dims[0] * channel + dims[1]) * height +
                                  dims[2]) *
                                     width +
                                 dims[3]];
                } else {
                  output_ptr[((n * channel + c) * height + h) * width + w] =
                      input_ptr[((n * channel + c) * height + h) * width + w] +
                      output_ptr[((dims[0] * channel + dims[1]) * height +
                                  dims[2]) *
                                     width +
                                 dims[3]];
                }
              }
            }
          }
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
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

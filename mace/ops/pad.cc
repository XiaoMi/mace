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
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/pad.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class PadOp;

template <typename T>
class PadOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit PadOp(OpConstructContext *context)
      : Operation(context),
        paddings_(Operation::GetRepeatedArgs<int>("paddings")),
        constant_value_(Operation::GetOptionalArg<float>(
            "constant_value", 0.0)) {
    MACE_CHECK(paddings_.size() == 8);
    auto df = static_cast<DataFormat>(Operation::GetOptionalArg<int>(
        "data_format", DataFormat::DF_NONE));
    if (df == DataFormat::NHWC) {
      paddings_ = TransposeShape<int, int>(paddings_, {0, 1, 6, 7, 2, 3, 4, 5});
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(
        this->paddings_.size() == static_cast<size_t>(input->dim_size()) * 2);
    auto input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->Resize({input_shape[0] + this->paddings_[0]
                                             + this->paddings_[1],
                                         input_shape[1] + this->paddings_[2]
                                             + this->paddings_[3],
                                         input_shape[2] + this->paddings_[4]
                                             + this->paddings_[5],
                                         input_shape[3] + this->paddings_[6]
                                             + this->paddings_[7]}));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();
    std::fill(output_ptr, output_ptr + output->size(), this->constant_value_);

    const index_t batch = input->dim(0);
    const index_t channel = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);
#pragma omp parallel for collapse(3)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        for (index_t h = 0; h < height; ++h) {
          const index_t in_offset = (((b * channel + c) * height) + h) * width;
          const index_t out_offset = (((b + this->paddings_[0]) * output->dim(1)
              + (c + this->paddings_[2])) * output->dim(2)
              + (h + this->paddings_[4])) * output->dim(3)
              + this->paddings_[6];
          memcpy(output_ptr + out_offset,
                 input_ptr + in_offset,
                 width * sizeof(T));
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> paddings_;
  float constant_value_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class PadOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit PadOp(OpConstructContext *context)
      : Operation(context) {
    std::vector<int> paddings = Operation::GetRepeatedArgs<int>("paddings");
    float constant_value = Operation::GetOptionalArg<float>(
        "constant_value", 0.0);
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::PadKernel<T>(paddings, constant_value));
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLPadKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterPad(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

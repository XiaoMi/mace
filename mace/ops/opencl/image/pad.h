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
#ifndef MACE_OPS_OPENCL_IMAGE_PAD_H_
#define MACE_OPS_OPENCL_IMAGE_PAD_H_

#include "mace/ops/opencl/pad.h"

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/pad.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class PadKernel : public OpenCLPadKernel {
 public:
  PadKernel(const PadType type,
            const std::vector<int> &paddings,
            const float constant_value)
      : type_(type), paddings_(paddings), constant_value_(constant_value) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  PadType type_;
  std::vector<int> paddings_;
  float constant_value_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus PadKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_CHECK(this->paddings_.size() ==
      static_cast<size_t>((input->dim_size() * 2)));
  MACE_CHECK((this->paddings_[0] == 0) && (this->paddings_[1] == 0) &&
      (this->paddings_[6] == 0) && (this->paddings_[7] == 0))
    << "Mace only support height/width dimension now";
  for (int i = 2; i <= 5; ++i) {
    MACE_CHECK(paddings_[i] >= 0);
  }
  auto input_shape = input->shape();
  if (type_ == PadType::REFLECT) {
    MACE_CHECK(paddings_[2] < input_shape[1] &&
               paddings_[3] < input_shape[1] &&
               paddings_[4] < input_shape[2] &&
               paddings_[5] < input_shape[2]);
  } else if (type_ == PadType::SYMMETRIC) {
    MACE_CHECK(paddings_[2] <= input_shape[1] &&
               paddings_[3] <= input_shape[1] &&
               paddings_[4] <= input_shape[2] &&
               paddings_[5] <= input_shape[2]);
  } else {
    MACE_CHECK(type_ == PadType::CONSTANT);
  }
  std::vector<index_t> output_shape = {
      input_shape[0] + this->paddings_[0] + this->paddings_[1],
      input_shape[1] + this->paddings_[2] + this->paddings_[3],
      input_shape[2] + this->paddings_[4] + this->paddings_[5],
      input_shape[3] + this->paddings_[6] + this->paddings_[7]};

  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pad");
    built_options.emplace("-Dpad=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(MakeString("-DPAD_TYPE=", type_));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("pad", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  MACE_OUT_OF_RANGE_INIT(kernel_);

  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    if (type_ == PadType::CONSTANT) {
      kernel_.setArg(idx++, this->constant_value_);
    }
    kernel_.setArg(idx++, static_cast<int32_t>(input_shape[1]));
    kernel_.setArg(idx++, static_cast<int32_t>(input_shape[2]));
    kernel_.setArg(idx++, static_cast<int32_t>(output_shape[1]));
    kernel_.setArg(idx++, this->paddings_[2]);
    kernel_.setArg(idx++, this->paddings_[4]);

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key = Concat("pad", output->dim(0), output->dim(1),
                                  output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_PAD_H_

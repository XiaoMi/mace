// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/kernels/pad.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus PadFunctor<DeviceType::GPU, T>::operator()(const Tensor *input,
                                                      Tensor *output,
                                                      StatsFuture *future) {
  MACE_CHECK(this->paddings_.size() ==
             static_cast<size_t>((input->dim_size() * 2)));
  MACE_CHECK((this->paddings_[0] == 0) && (this->paddings_[1] == 0) &&
             (this->paddings_[6] == 0) && (this->paddings_[7] == 0))
      << "Mace only support height/width dimension now";
  auto input_shape = input->shape();
  std::vector<index_t> output_shape = {
      input_shape[0] + this->paddings_[0] + this->paddings_[1],
      input_shape[1] + this->paddings_[2] + this->paddings_[3],
      input_shape[2] + this->paddings_[4] + this->paddings_[5],
      input_shape[3] + this->paddings_[6] + this->paddings_[7]};

  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = context_->device()->opencl_runtime();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_, context_);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pad");
    built_options.emplace("-Dpad=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("pad", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, this->constant_value_);
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
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct PadFunctor<DeviceType::GPU, float>;
template struct PadFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

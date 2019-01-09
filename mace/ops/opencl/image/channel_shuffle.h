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
#ifndef MACE_OPS_OPENCL_IMAGE_CHANNEL_SHUFFLE_H_
#define MACE_OPS_OPENCL_IMAGE_CHANNEL_SHUFFLE_H_

#include "mace/ops/opencl/channel_shuffle.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class ChannelShuffleKernel : public OpenCLChannelShuffleKernel {
 public:
  explicit ChannelShuffleKernel(const int groups) : groups_(groups) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  const int groups_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus ChannelShuffleKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_CHECK(input->dim(3) % groups_ == 0,
             "input channels must be an integral multiple of group. ",
             input->dim(3));
  MACE_RETURN_IF_ERROR(output->ResizeLike(input));

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);
  const index_t channels_per_group = channels / groups_;
  const index_t group_channel_blocks = RoundUpDiv4(channels_per_group);

  const uint32_t gws[3] = {static_cast<uint32_t>(group_channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();

  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("channel_shuffle");
    built_options.emplace("-Dchannel_shuffle=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(
        runtime->BuildKernel("channel_shuffle", kernel_name,
                             built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, groups_);
    kernel_.setArg(idx++, static_cast<uint32_t>(channels_per_group));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("channel_shuffle_opencl_kernel", output->dim(0), output->dim(1),
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

#endif  // MACE_OPS_OPENCL_IMAGE_CHANNEL_SHUFFLE_H_

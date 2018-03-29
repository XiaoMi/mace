//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/channel_shuffle.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void ChannelShuffleFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    Tensor *output,
    StatsFuture *future) {
  output->ResizeLike(input);

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);
  const index_t channels_per_group = channels / groups_;
  MACE_CHECK(channels_per_group % 4 == 0,
             "channels per group must be multiple of 4");
  MACE_CHECK(groups_ % 4 == 0,
             "groups must be multiple of 4");
  const index_t group_channel_blocks = RoundUpDiv4(channels_per_group);

  const uint32_t gws[3] = {static_cast<uint32_t>(group_channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    is_non_uniform_work_groups_supported_ =
        runtime->IsNonUniformWorkgroupsSupported();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("channel_shuffle");
    built_options.emplace("-Dchannel_shuffle=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    if (is_non_uniform_work_groups_supported_) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }
    kernel_ = runtime->BuildKernel("channel_shuffle", kernel_name,
                                   built_options);
  }

  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    if (!is_non_uniform_work_groups_supported_) {
      kernel_.setArg(idx++, gws[0]);
      kernel_.setArg(idx++, gws[1]);
      kernel_.setArg(idx++, gws[2]);
    }
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, groups_);
    kernel_.setArg(idx++, static_cast<uint32_t>(channels_per_group));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const std::vector<uint32_t> lws = {8, kwg_size_ / 64, 8, 1};
  std::stringstream ss;
  ss << "channel_shuffle_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
}

template
struct ChannelShuffleFunctor<DeviceType::OPENCL, float>;
template
struct ChannelShuffleFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

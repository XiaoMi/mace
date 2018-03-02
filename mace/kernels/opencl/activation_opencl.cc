//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/activation.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void ActivationFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  if (kernel_.get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();

    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("activation");
    built_options.emplace("-Dactivation=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    switch (activation_) {
      case RELU:
        tuning_key_prefix_ = "relu_opencl_kernel_";
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        tuning_key_prefix_ = "relux_opencl_kernel_";
        built_options.emplace("-DUSE_RELUX");
        break;
      case PRELU:
        tuning_key_prefix_ = "prelu_opencl_kernel_";
        built_options.emplace("-DUSE_PRELU");
        break;
      case TANH:
        tuning_key_prefix_ = "tanh_opencl_kernel_";
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        tuning_key_prefix_ = "sigmoid_opencl_kernel_";
        built_options.emplace("-DUSE_SIGMOID");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    kernel_ =
        runtime->BuildKernel("activation", kernel_name, built_options);
    int idx = 0;
    kernel_.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
    kernel_.setArg(idx++, static_cast<float>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<float>(prelu_alpha_));
    kernel_.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::string tuning_key =
      Concat(tuning_key_prefix_, output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  TuningOrRun3DKernel(kernel_, tuning_key, gws, lws, future);
}

template struct ActivationFunctor<DeviceType::OPENCL, float>;
template struct ActivationFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

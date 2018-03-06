//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/eltwise.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void EltwiseFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input0,
                                                       const Tensor *input1,
                                                       Tensor *output,
                                                       StatsFuture *future) {

  const index_t batch = input0->dim(0);
  const index_t height = input0->dim(1);
  const index_t width = input0->dim(2);
  const index_t channels = input0->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  if (kernel_.get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (!coeff_.empty()) built_options.emplace("-DCOEFF_SUM");
    kernel_ = runtime->BuildKernel("eltwise", kernel_name, built_options);

    uint32_t idx = 0;
    kernel_.setArg(idx++,
                   *(input0->opencl_image()));
    kernel_.setArg(idx++,
                   *(input1->opencl_image()));
    if (!coeff_.empty()) {
      kernel_.setArg(idx++, coeff_[0]);
      kernel_.setArg(idx++, coeff_[1]);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_pixels),
      static_cast<uint32_t>(batch_height_pixels)
  };
  const std::vector<uint32_t> lws = {64, 16, 1};
  std::stringstream ss;
  ss << "eltwise_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct EltwiseFunctor<DeviceType::OPENCL, float>;
template struct EltwiseFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

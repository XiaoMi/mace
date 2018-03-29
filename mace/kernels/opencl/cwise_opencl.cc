//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/cwise.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void CWiseFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                       Tensor *output,
                                                       StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  if (kernel_.get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("cwise");
    built_options.emplace("-Dcwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DCWISE_TYPE=", type_));
    kernel_ = runtime->BuildKernel("cwise", kernel_name, built_options);
  }
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<float>(coeff_));
    kernel_.setArg(idx++, *(output->opencl_image()));
    input_shape_ = input->shape();
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(width_pixels),
                           static_cast<uint32_t>(batch_height_pixels)};
  const std::vector<uint32_t> lws = {64, 16, 1};
  std::stringstream ss;
  ss << "cwise_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct CWiseFunctor<DeviceType::OPENCL, float>;
template struct CWiseFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

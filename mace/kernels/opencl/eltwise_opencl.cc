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

  const uint32_t gws[2] = {static_cast<uint32_t>(width_pixels),
                           static_cast<uint32_t>(batch_height_pixels)};

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    is_non_uniform_work_groups_supported_ =
        runtime->IsNonUniformWorkgroupsSupported();
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (is_non_uniform_work_groups_supported_) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }
    if (!coeff_.empty()) built_options.emplace("-DCOEFF_SUM");
    kernel_ = runtime->BuildKernel("eltwise", kernel_name, built_options);
  }
  if (!IsVecEqual(input_shape_, input0->shape())) {
    uint32_t idx = 0;
    if (!is_non_uniform_work_groups_supported_) {
      kernel_.setArg(idx++, gws[0]);
      kernel_.setArg(idx++, gws[1]);
    }
    kernel_.setArg(idx++, *(input0->opencl_image()));
    kernel_.setArg(idx++, *(input1->opencl_image()));
    if (!coeff_.empty()) {
      kernel_.setArg(idx++, coeff_[0]);
      kernel_.setArg(idx++, coeff_[1]);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input0->shape();

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 16, 16, 1};
  std::stringstream ss;
  ss << "eltwise_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct EltwiseFunctor<DeviceType::OPENCL, float>;
template struct EltwiseFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

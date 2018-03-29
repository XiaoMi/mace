//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void AddNFunctor<DeviceType::OPENCL, T>::operator()(
    const std::vector<const Tensor *> &input_tensors,
    Tensor *output_tensor,
    StatsFuture *future) {
  size_t size = input_tensors.size();
  MACE_CHECK(size >= 2 && input_tensors[0] != nullptr);

  const index_t batch = input_tensors[0]->dim(0);
  const index_t height = input_tensors[0]->dim(1);
  const index_t width = input_tensors[0]->dim(2);
  const index_t channels = input_tensors[0]->dim(3);

  auto runtime = OpenCLRuntime::Global();

  for (int i = 1; i < size; ++i) {
    MACE_CHECK_NOTNULL(input_tensors[i]);
    MACE_CHECK(batch == input_tensors[i]->dim(0));
    MACE_CHECK(height == input_tensors[i]->dim(1));
    MACE_CHECK(width == input_tensors[i]->dim(2));
    MACE_CHECK(channels == input_tensors[i]->dim(3));
  }

  if (kernel_.get() == nullptr) {
    is_non_uniform_work_groups_supported_ =
        runtime->IsNonUniformWorkgroupsSupported();
    if (input_tensors.size() > 4) {
      MACE_NOT_IMPLEMENTED;
    }
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("addn");
    built_options.emplace("-Daddn=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DINPUT_NUM=", input_tensors.size()));
    if (is_non_uniform_work_groups_supported_) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }

    kernel_ = runtime->BuildKernel("addn", kernel_name, built_options);
  }

  std::vector<index_t> output_shape = input_tensors[0]->shape();

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  const uint32_t gws[2] = {static_cast<uint32_t>(width_pixels),
                           static_cast<uint32_t>(batch_height_pixels)};

  if (!IsVecEqual(input_shape_, input_tensors[0]->shape())) {
    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                    &output_image_shape);
    output_tensor->ResizeImage(output_shape, output_image_shape);

    uint32_t idx = 0;
    if (!is_non_uniform_work_groups_supported_) {
      kernel_.setArg(idx++, gws[0]);
      kernel_.setArg(idx++, gws[1]);
    }
    for (auto input : input_tensors) {
      kernel_.setArg(idx++, *(input->opencl_image()));
    }
    kernel_.setArg(idx++, *(output_tensor->opencl_image()));

    input_shape_ = input_tensors[0]->shape();

    kwg_size_ =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 16, 16, 1};
  std::stringstream ss;
  ss << "addn_opencl_kernel_" << output_shape[0] << "_" << output_shape[1]
     << "_" << output_shape[2] << "_" << output_shape[3];
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct AddNFunctor<DeviceType::OPENCL, float>;

template struct AddNFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

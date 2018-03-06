//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

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

  for (int i = 1; i < size; ++i) {
    MACE_CHECK_NOTNULL(input_tensors[i]);
    MACE_CHECK(batch == input_tensors[i]->dim(0));
    MACE_CHECK(height == input_tensors[i]->dim(1));
    MACE_CHECK(width == input_tensors[i]->dim(2));
    MACE_CHECK(channels == input_tensors[i]->dim(3));
  }

  std::vector<index_t> output_shape = input_tensors[0]->shape();
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, output_image_shape);
  output_tensor->ResizeImage(output_shape, output_image_shape);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  if (kernel_.get() == nullptr) {
    if (input_tensors.size() > 4) {
      MACE_NOT_IMPLEMENTED;
    }
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("addn");
    built_options.emplace("-Daddn=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DINPUT_NUM=", input_tensors.size()));
    kernel_ = runtime->BuildKernel("addn", kernel_name, built_options);

    uint32_t idx = 0;
    for (auto input : input_tensors) {
      kernel_.setArg(idx++,
                         *(input->opencl_image()));
    }
    kernel_.setArg(idx++, *(output_tensor->opencl_image()));
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_pixels),
      static_cast<uint32_t>(batch_height_pixels)
  };
  const std::vector<uint32_t> lws = {64, 16, 1};
  std::stringstream ss;
  ss << "addn_opencl_kernel_"
     << output_shape[0] << "_"
     << output_shape[1] << "_"
     << output_shape[2] << "_"
     << output_shape[3];
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
};

template
struct AddNFunctor<DeviceType::OPENCL, float>;

template
struct AddNFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

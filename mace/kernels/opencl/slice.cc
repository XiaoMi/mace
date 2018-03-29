//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/slice.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template<typename T>
void SliceFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    const std::vector<Tensor *> &output_list,
    StatsFuture *future) {
  const index_t input_channels = input->dim(3);
  const size_t outputs_count = output_list.size();
  const index_t output_channels = input_channels / outputs_count;
  MACE_CHECK(output_channels % 4 == 0)
    << "output channels of slice op must be divisible by 4";
  std::vector<index_t> output_shape({input->dim(0), input->dim(1),
                                     input->dim(2), output_channels});

  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  for (size_t i= 0; i < outputs_count; ++i) {
    output_list[i]->ResizeImage(output_shape, image_shape);
  }

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("slice");
    built_options.emplace("-Dslice=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE="
                           + DtToCLCMDDt(DataTypeToEnum<T>::value));
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
    kernel_ = runtime->BuildKernel("slice", kernel_name, built_options);

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  const index_t channel_blk = RoundUpDiv4(output_channels);

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk),
      static_cast<uint32_t>(input->dim(2)),
      static_cast<uint32_t>(input->dim(0) * input->dim(1)),
  };

  const std::vector<uint32_t> lws = {8, kwg_size_ / 64, 8, 1};
  std::stringstream ss;
  ss << "slice_opencl_kernel_"
     << input->dim(0) << "_"
     << input->dim(1) << "_"
     << input->dim(2) << "_"
     << input_channels << "_"
     << outputs_count;
  for (int i = 0; i < outputs_count; ++i) {
    uint32_t idx = 0;
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      kernel_.setArg(idx++, gws[0]);
      kernel_.setArg(idx++, gws[1]);
      kernel_.setArg(idx++, gws[2]);
    }
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blk * i));
    kernel_.setArg(idx++, *(output_list[i]->opencl_image()));

    TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
  }
}

template
struct SliceFunctor<DeviceType::OPENCL, float>;
template
struct SliceFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

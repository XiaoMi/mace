//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/kernels/resize_bilinear.h"

namespace mace {
namespace kernels {

template <>
void ResizeBilinearFunctor<DeviceType::OPENCL, float>::operator()(
    const Tensor *input, const Tensor *resize_dims, Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t in_height = input->dim(2);
  const index_t in_width = input->dim(3);

  index_t out_height;
  index_t out_width;
  GetOutputSize(resize_dims, &out_height, &out_width);
  MACE_CHECK(out_height > 0 && out_width > 0);
  std::vector<index_t> out_shape {batch, channels, out_height, out_width};
  output->Resize(out_shape);

  float height_scale =
      CalculateResizeScale(in_height, out_height, align_corners_);
  float width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  auto rb_kernel  = runtime->BuildKernel("resize_bilinear", "resize_bilinear_nocache", built_options);

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(rb_kernel);
  uint32_t idx = 0;
  rb_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
  rb_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));
  rb_kernel.setArg(idx++, height_scale);
  rb_kernel.setArg(idx++, width_scale);
  rb_kernel.setArg(idx++, static_cast<int>(in_height));
  rb_kernel.setArg(idx++, static_cast<int>(in_width));

  auto command_queue = runtime->command_queue();

  cl_int error = command_queue.enqueueNDRangeKernel(
      rb_kernel, cl::NullRange,
      cl::NDRange(static_cast<int>(batch * channels),
                  static_cast<int>(out_height), static_cast<int>(out_width)),
      // TODO (heliangliang) tuning and fix when kwg_size < devisor
      cl::NDRange(1, 16, kwg_size / 16),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);
}

}  // namespace kernels
}  // namespace mace

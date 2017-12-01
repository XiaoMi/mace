//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/kernels/resize_bilinear.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void ResizeBilinearFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input, const Tensor *resize_dims, Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  index_t out_height;
  index_t out_width;
  GetOutputSize(resize_dims, &out_height, &out_width);
  MACE_CHECK(out_height > 0 && out_width > 0);
  std::vector<index_t> output_shape {batch, out_height, out_width, channels};
  if (input->is_image()) {
    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
    output->ResizeImage(output_shape, output_image_shape);
  } else {
    output->Resize(output_shape);
  }

  float height_scale =
      CalculateResizeScale(in_height, out_height, align_corners_);
  float width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DataTypeToOPENCLCMDDataType(dt));
  auto rb_kernel  = runtime->BuildKernel("resize_bilinear", "resize_bilinear_nocache", built_options);

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(rb_kernel);

  uint32_t idx = 0;
  rb_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  rb_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  rb_kernel.setArg(idx++, height_scale);
  rb_kernel.setArg(idx++, width_scale);
  rb_kernel.setArg(idx++, static_cast<int32_t>(in_height));
  rb_kernel.setArg(idx++, static_cast<int32_t>(in_width));
  rb_kernel.setArg(idx++, static_cast<int32_t>(out_height));

  auto command_queue = runtime->command_queue();

  cl_int error = command_queue.enqueueNDRangeKernel(
      rb_kernel, cl::NullRange,
      cl::NDRange(static_cast<int32_t>(channel_blocks),
                  static_cast<int32_t>(out_width),
                  static_cast<int32_t>(out_height * batch)),
      // TODO tuning
      cl::NDRange(1, static_cast<int32_t>(out_width > kwg_size ? kwg_size : out_width), 1),
      nullptr, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);
}

template struct ResizeBilinearFunctor<DeviceType::OPENCL, float>;
template struct ResizeBilinearFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

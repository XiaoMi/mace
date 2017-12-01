//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/buffer_to_image.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

template<typename T>
void BufferToImageFunctor<DeviceType::OPENCL, T>::operator()(Tensor *buffer,
                                                             const BufferType type,
                                                             Tensor *image) {
  MACE_CHECK(!buffer->is_image()) << "buffer must be buffer-type";
  std::vector<size_t> image_shape;
  if (!i2b_) {
    CalImage2DShape(buffer->shape(), type, image_shape);
    image->ResizeImage(buffer->shape(), image_shape);
  } else {
    image_shape = image->image_shape();
    buffer->Resize(image->shape());
  }

  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(DataTypeToEnum<T>::value));
  built_options.emplace("-DCMD_DATA_TYPE=" + DataTypeToOPENCLCMDDataType(DataTypeToEnum<T>::value));
  auto runtime = OpenCLRuntime::Get();
  string kernel_name;
  switch (type) {
    case FILTER:
      kernel_name = i2b_ ? "filter_image_to_buffer" : "filter_buffer_to_image";
      break;
    case IN_OUT:
      kernel_name = i2b_ ? "in_out_image_to_buffer" : "in_out_buffer_to_image";
      break;
    case ARGUMENT:
      kernel_name = i2b_ ? "arg_image_to_buffer" : "arg_buffer_to_image";
      break;
  }
  auto b2f_kernel = runtime->BuildKernel("buffer_to_image",
                                         kernel_name,
                                         built_options);

  uint32_t idx = 0;
  b2f_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(buffer->buffer())));
  if (type == ARGUMENT) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
  } else {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
  }
  b2f_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(image->buffer())));

  const size_t gws[3] = {image_shape[0],
                         image_shape[1],
                         1};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(b2f_kernel);
  const std::vector<uint32_t> lws = {kwg_size, 1, 1};
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      b2f_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]));

  MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
}

template struct BufferToImageFunctor<DeviceType::OPENCL, float>;
template struct BufferToImageFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

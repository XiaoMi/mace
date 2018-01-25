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
                                                             Tensor *image,
                                                             StatsFuture *future) {
  MACE_CHECK(!buffer->is_image()) << "buffer must be buffer-type";
  std::vector<size_t> image_shape;
  if (!i2b_) {
    CalImage2DShape(buffer->shape(), type, image_shape);
    image->ResizeImage(buffer->shape(), image_shape);
    buffer->MarkUnused();
  } else {
    image_shape = image->image_shape();
    buffer->Resize(image->shape());
  }

  string kernel_name;
  switch (type) {
    case CONV2D_FILTER:
      kernel_name = i2b_ ? "filter_image_to_buffer" : "filter_buffer_to_image";
      break;
    case DW_CONV2D_FILTER:
      kernel_name = i2b_ ? "dw_filter_image_to_buffer" : "dw_filter_buffer_to_image";
      break;
    case IN_OUT:
      kernel_name = i2b_ ? "in_out_image_to_buffer" : "in_out_buffer_to_image";
      break;
    case ARGUMENT:
      kernel_name = i2b_ ? "arg_image_to_buffer" : "arg_buffer_to_image";
      break;
  }
  string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  if (buffer->dtype() == image->dtype()) {
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DataTypeToEnum<T>::value));
  } else {
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
  }
  auto runtime = OpenCLRuntime::Global();
  auto b2f_kernel = runtime->BuildKernel("buffer_to_image",
                                         obfuscated_kernel_name,
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
  const std::vector<uint32_t> lws = {16, 64, 1};
  cl::Event event;
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      b2f_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      nullptr, &event);
  MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;

  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
}

template struct BufferToImageFunctor<DeviceType::OPENCL, float>;
template struct BufferToImageFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/buffer_to_image.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

template <typename T>
void BufferToImageFunctor<DeviceType::OPENCL, T>::operator()(
    Tensor *buffer, const BufferType type, Tensor *image, StatsFuture *future) {
  std::vector<size_t> image_shape;
  if (!i2b_) {
    CalImage2DShape(buffer->shape(), type, &image_shape);
    if (type == WINOGRAD_FILTER) {
      std::vector<index_t> new_shape = CalWinogradShape(buffer->shape(), type);
      image->ResizeImage(new_shape, image_shape);
    } else {
      image->ResizeImage(buffer->shape(), image_shape);
    }
  } else {
    CalImage2DShape(image->shape(), type, &image_shape);
    buffer->Resize(image->shape());
  }

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
    case CONV2D_FILTER:
      kernel_name = i2b_ ? "filter_image_to_buffer" : "filter_buffer_to_image";
      break;
    case DW_CONV2D_FILTER:
      kernel_name =
          i2b_ ? "dw_filter_image_to_buffer" : "dw_filter_buffer_to_image";
      break;
    case IN_OUT_CHANNEL:
      kernel_name = i2b_ ? "in_out_image_to_buffer" : "in_out_buffer_to_image";
      break;
    case ARGUMENT:
      kernel_name = i2b_ ? "arg_image_to_buffer" : "arg_buffer_to_image";
      break;
    case IN_OUT_HEIGHT:
    case WEIGHT_HEIGHT:
      kernel_name = i2b_ ? "in_out_height_image_to_buffer"
                         : "in_out_height_buffer_to_image";
      break;
    case IN_OUT_WIDTH:
    case WEIGHT_WIDTH:
      MACE_CHECK(!i2b_) << "IN_OUT_WIDTH only support buffer to image now";
      kernel_name = "in_out_width_buffer_to_image";
      break;
    case WINOGRAD_FILTER:
      gws[1] /= 16;
      kernel_name = i2b_ ? "winograd_filter_image_to_buffer"
                         : "winograd_filter_buffer_to_image";
      break;
  }

  auto runtime = OpenCLRuntime::Global();

  const bool is_non_uniform_work_groups_supported =
      runtime->IsNonUniformWorkgroupsSupported();

  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  if (is_non_uniform_work_groups_supported) {
    built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
  }
  if (buffer->dtype() == image->dtype()) {
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(DataTypeToEnum<T>::value));
  } else {
    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpstreamCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
  }
  auto b2f_kernel = runtime->BuildKernel("buffer_to_image",
                                         obfuscated_kernel_name, built_options);

  uint32_t idx = 0;
  b2f_kernel.setArg(idx++, *(buffer->opencl_buffer()));
  if (!i2b_) {
    MACE_CHECK(buffer->buffer_offset() % GetEnumTypeSize(buffer->dtype()) == 0,
               "buffer offset not aligned");
    b2f_kernel.setArg(idx++,
                      static_cast<uint32_t>(buffer->buffer_offset() /
                                            GetEnumTypeSize(buffer->dtype())));
  }
  if (type == CONV2D_FILTER) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
  } else if (type == ARGUMENT) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
  } else if (type == WEIGHT_HEIGHT || type == WEIGHT_WIDTH) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
    b2f_kernel.setArg(idx++, 1);
  } else {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
  }
  b2f_kernel.setArg(idx++, *(image->opencl_image()));
  b2f_kernel.setArg(idx++, gws[0]);
  b2f_kernel.setArg(idx++, gws[1]);

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(b2f_kernel));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  cl_int error;
  if (is_non_uniform_work_groups_supported) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        b2f_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }

    error = runtime->command_queue().enqueueNDRangeKernel(
        b2f_kernel, cl::NullRange, cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  }
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

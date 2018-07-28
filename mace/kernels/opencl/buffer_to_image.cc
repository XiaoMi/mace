// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/kernels/buffer_to_image.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus BufferToImageFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *buffer,
    const BufferType type,
    Tensor *image,
    StatsFuture *future) {
  auto formatted_buffer_shape = FormatBufferShape(buffer->shape(), type);
  std::vector<size_t> image_shape;
  CalImage2DShape(formatted_buffer_shape, type, &image_shape, wino_blk_size_);
  if (type == WINOGRAD_FILTER) {
    std::vector<index_t> new_shape =
        CalWinogradShape(buffer->shape(), type, wino_blk_size_);
    MACE_RETURN_IF_ERROR(image->ResizeImage(new_shape, image_shape));
  } else {
    MACE_RETURN_IF_ERROR(image->ResizeImage(buffer->shape(), image_shape));
  }

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
    case CONV2D_FILTER:
      kernel_name = "filter_buffer_to_image";
      break;
    case DW_CONV2D_FILTER:
      kernel_name = "dw_filter_buffer_to_image";
      break;
    case IN_OUT_CHANNEL:
      kernel_name = "in_out_buffer_to_image";
      break;
    case ARGUMENT:
      kernel_name = "arg_buffer_to_image";
      break;
    case IN_OUT_HEIGHT:
      kernel_name = "in_out_height_buffer_to_image";
      break;
    case IN_OUT_WIDTH:
      kernel_name = "in_out_width_buffer_to_image";
      break;
    case WEIGHT_HEIGHT:
      kernel_name = "weight_height_buffer_to_image";
      break;
    case WEIGHT_WIDTH:
      kernel_name = "weight_width_buffer_to_image";
      break;
    case WINOGRAD_FILTER: {
      std::stringstream ss_tmp;
      gws[1] /= (wino_blk_size_ + 2) * (wino_blk_size_ + 2);
      ss_tmp << "winograd_filter_buffer_to_image_"
             << wino_blk_size_ << "x" << wino_blk_size_;
      kernel_name = ss_tmp.str();
      break;
    }
  }

  auto runtime = OpenCLRuntime::Global();

  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
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
  if (runtime->IsOutOfRangeCheckEnabled()) {
    built_options.emplace("-DOUT_OF_RANGE_CHECK");
    if (!kernel_error_) {
      kernel_error_ = std::move(std::unique_ptr<Buffer>(
          new Buffer(GetDeviceAllocator(DeviceType::GPU))));
      MACE_RETURN_IF_ERROR(kernel_error_->Allocate(1));
      kernel_error_->Map(nullptr);
      *(kernel_error_->mutable_data<char>()) = 0;
      kernel_error_->UnMap();
    }
  }

  cl::Kernel b2f_kernel;

  MACE_RETURN_IF_ERROR(runtime->BuildKernel(
      "buffer_to_image", obfuscated_kernel_name, built_options, &b2f_kernel));

  uint32_t idx = 0;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    b2f_kernel.setArg(idx++,
                      *(static_cast<cl::Buffer *>(kernel_error_->buffer())));
  }
  if (!runtime->IsNonUniformWorkgroupsSupported()) {
    b2f_kernel.setArg(idx++, gws[0]);
    b2f_kernel.setArg(idx++, gws[1]);
  }
  b2f_kernel.setArg(idx++, *(buffer->opencl_buffer()));
  MACE_CHECK(buffer->buffer_offset() % GetEnumTypeSize(buffer->dtype()) == 0,
             "buffer offset not aligned");
  b2f_kernel.setArg(idx++,
                    static_cast<uint32_t>(buffer->buffer_offset() /
                                          GetEnumTypeSize(buffer->dtype())));
  if (type == CONV2D_FILTER) {
    const index_t inner_size = buffer->dim(1) * buffer->dim(2) * buffer->dim(3);
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(inner_size));
  } else if (type == DW_CONV2D_FILTER || type == WEIGHT_HEIGHT) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
  } else if (type == ARGUMENT) {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
  } else {
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[1]));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[2]));
    b2f_kernel.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[3]));
  }
  b2f_kernel.setArg(idx++, *(image->opencl_image()));

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(b2f_kernel));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
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
  MACE_CL_RET_STATUS(error);
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_error_->Map(nullptr);
    char *kerror_code = kernel_error_->mutable_data<char>();
    MACE_CHECK(*kerror_code == 0) << "Kernel error code: " << *kerror_code;
    kernel_error_->UnMap();
  }
  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

  // Mark the buffer unused.
  const_cast<Tensor *>(buffer)->MarkUnused();

  return MACE_SUCCESS;
}

template struct BufferToImageFunctor<DeviceType::GPU, float>;
template struct BufferToImageFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

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

#include "mace/kernels/image_to_buffer.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus ImageToBufferFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *image,
    const BufferType type,
    Tensor *buffer,
    StatsFuture *future) {
  auto formatted_buffer_shape = FormatBufferShape(image->shape(), type);
  std::vector<size_t> image_shape;
  CalImage2DShape(formatted_buffer_shape, type, &image_shape, wino_blk_size_);
  MACE_RETURN_IF_ERROR(buffer->Resize(image->shape()));

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
    case CONV2D_FILTER:
      kernel_name = "filter_image_to_buffer";
      break;
    case IN_OUT_CHANNEL:
      kernel_name = "in_out_image_to_buffer";
      break;
    case ARGUMENT:
      kernel_name = "arg_image_to_buffer";
      break;
    case IN_OUT_HEIGHT:
      kernel_name = "in_out_height_image_to_buffer";
      break;
    case WINOGRAD_FILTER: {
      std::stringstream ss_tmp;
      gws[1] /= (wino_blk_size_ + 2) * (wino_blk_size_ + 2);
      ss_tmp << "winograd_filter_image_to_buffer_"
             << wino_blk_size_ << "x" << wino_blk_size_;
      kernel_name = ss_tmp.str();
      break;
    }
    case WEIGHT_HEIGHT:
      kernel_name = "weight_height_image_to_buffer";
      break;
    case WEIGHT_WIDTH:
      kernel_name = "weight_width_image_to_buffer";
      break;
    case DW_CONV2D_FILTER:
    case IN_OUT_WIDTH:
      LOG(FATAL) << "IN_OUT_WIDTH only support buffer to image now";
      break;
  }

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    if (buffer->dtype() == image->dtype()) {
      built_options.emplace(
          "-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
      built_options.emplace("-DCMD_DATA_TYPE=" +
          DtToCLCMDDt(DataTypeToEnum<T>::value));
    } else {
      built_options.emplace("-DDATA_TYPE=" +
          DtToUpCompatibleCLDt(DataTypeToEnum<T>::value));
      built_options.emplace("-DCMD_DATA_TYPE=" +
          DtToUpCompatibleCLCMDDt(DataTypeToEnum<T>::value));
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_to_image",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));
  }

  if (!IsVecEqual(input_shape_, image->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_2D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(buffer->opencl_buffer()));
    if (type == CONV2D_FILTER) {
      const index_t
          inner_size = buffer->dim(1) * buffer->dim(2) * buffer->dim(3);
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
      kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
    } else if (type == WEIGHT_HEIGHT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(buffer->dim(3)));
    } else {
      kernel_.setArg(idx++,
                        static_cast<uint32_t>(formatted_buffer_shape[1]));
      kernel_.setArg(idx++,
                        static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++,
                        static_cast<uint32_t>(formatted_buffer_shape[3]));
    }
    kernel_.setArg(idx++, *(image->opencl_image()));
    input_shape_ = image->shape();
  }

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  const std::vector<uint32_t> lws = {16, kwg_size / 16};

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }

    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(roundup_gws[0], roundup_gws[1]),
        cl::NDRange(lws[0], lws[1]), nullptr, &event);
  }
  MACE_CL_RET_STATUS(error);
  OUT_OF_RANGE_VALIDATION(kernel_error_);
  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

  return MACE_SUCCESS;
}

template struct ImageToBufferFunctor<DeviceType::GPU, float>;
template struct ImageToBufferFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

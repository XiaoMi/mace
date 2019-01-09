// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_OPENCL_IMAGE_IMAGE_TO_BUFFER_H_
#define MACE_OPS_OPENCL_IMAGE_IMAGE_TO_BUFFER_H_

#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/ops/opencl/buffer_transform_kernel.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class ImageToBuffer : public OpenCLBufferTransformKernel {
 public:
  MaceStatus Compute(OpContext *context,
                     const Tensor *input,
                     const OpenCLBufferType type,
                     const int wino_blk_size,
                     Tensor *output) override;

 private:
  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus ImageToBuffer<T>::Compute(OpContext *context,
                                     const Tensor *input,
                                     const OpenCLBufferType type,
                                     const int wino_blk_size,
                                     Tensor *output) {
  auto formatted_buffer_shape = FormatBufferShape(input->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(formatted_buffer_shape,
                              type,
                              &image_shape,
                              wino_blk_size);
  MACE_RETURN_IF_ERROR(output->Resize(input->shape()));

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
      gws[1] /= (wino_blk_size + 2) * (wino_blk_size + 2);
      ss_tmp << "winograd_filter_image_to_buffer_"
             << wino_blk_size << "x" << wino_blk_size;
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

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    if (output->dtype() == input->dtype()) {
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

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    if (type == CONV2D_FILTER) {
      const index_t
          inner_size = output->dim(1) * output->dim(2) * output->dim(3);
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(3)));
      kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
    } else if (type == WEIGHT_HEIGHT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(output->dim(3)));
    } else {
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[1]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++,
                     static_cast<uint32_t>(formatted_buffer_shape[3]));
    }
    kernel_.setArg(idx++, *(input->opencl_image()));
    input_shape_ = input->shape();
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
  MACE_OUT_OF_RANGE_VALIDATION;
  if (context->future() != nullptr) {
    context->future()->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_IMAGE_TO_BUFFER_H_

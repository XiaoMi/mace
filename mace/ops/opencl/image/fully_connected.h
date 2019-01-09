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
#ifndef MACE_OPS_OPENCL_IMAGE_FULLY_CONNECTED_H_
#define MACE_OPS_OPENCL_IMAGE_FULLY_CONNECTED_H_

#include "mace/ops/opencl/fully_connected.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class FullyConnectedKernel : public OpenCLFullyConnectedKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *weight,
      const Tensor *bias,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  std::vector<uint32_t> gws_;
  std::vector<uint32_t> lws_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus FullyConnectedKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    const ActivationType activation,
    const float relux_max_limit,
    const float leakyrelu_coefficient,
    Tensor *output) {
  std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    const index_t batch = output->dim(0);
    const index_t output_size = output->dim(3);
    const index_t output_blocks = RoundUpDiv4(output_size);

    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected_width");
    built_options.emplace("-Dfully_connected_width=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    if (bias != nullptr) {
      built_options.emplace("-DBIAS");
    }
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }
    if (runtime->gpu_type() != GPUType::QUALCOMM_ADRENO) {
      built_options.emplace("-DNON_QUALCOMM_ADRENO");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("fully_connected", kernel_name,
                                              built_options, &kernel_));

    const uint32_t kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));

    if (runtime->gpu_type() == GPUType::QUALCOMM_ADRENO) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
      const uint32_t wave_size =
          static_cast<uint32_t>(runtime->GetKernelWaveSize(kernel_));

      gws_ = {4, (wave_size / 4), static_cast<uint32_t>(batch * output_blocks)};

      const uint32_t inter_local_blks = kwg_size / (gws_[0] * gws_[1]);
      lws_ = {gws_[0], gws_[1], inter_local_blks};
    } else {
      gws_ = {4, 8, static_cast<uint32_t>(batch * output_blocks)};

      const uint32_t inter_local_blks = kwg_size / (gws_[0] * gws_[1]);
      lws_ = {gws_[0], gws_[1], inter_local_blks};
    }
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    const index_t batch = output->dim(0);
    const index_t output_blocks = RoundUpDiv4(output->dim(3));
    gws_[2] = static_cast<uint32_t>(batch * output_blocks);

    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(weight->opencl_image()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_image()));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, (lws_[0] * lws_[1] * lws_[2] * sizeof(float)),
                   nullptr);
    kernel_.setArg(idx++, static_cast<int>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(input->dim(3))));
    kernel_.setArg(idx++, static_cast<int>(output_blocks));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);

    input_shape_ = input->shape();
  }
  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws_[0], gws_[1], gws_[2]),
        cl::NDRange(lws_[0], lws_[1], lws_[2]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws_.size());
    for (size_t i = 0; i < lws_.size(); ++i) {
      roundup_gws[i] = RoundUp(gws_[i], lws_[i]);
    }
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange,
        cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
        cl::NDRange(lws_[0], lws_[1], lws_[2]), nullptr, &event);
  }
  MACE_OUT_OF_RANGE_VALIDATION;
  MACE_CL_RET_STATUS(error);

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

#endif  // MACE_OPS_OPENCL_IMAGE_FULLY_CONNECTED_H_

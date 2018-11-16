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
#ifndef MACE_OPS_OPENCL_IMAGE_REDUCE_MEAN_H_
#define MACE_OPS_OPENCL_IMAGE_REDUCE_MEAN_H_

#include "mace/ops/opencl/reduce_mean.h"

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
class ReduceMeanKernel : public OpenCLReduceMeanKernel {
 public:
  ReduceMeanKernel(const std::vector<int> axis,
                   const bool keep_dims)
      : axis_(axis), keep_dims_(keep_dims) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  const std::vector<int> axis_;
  bool keep_dims_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus ReduceMeanKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK(keep_dims_, "reduce mean gpu only support keep dims.");
  MACE_CHECK(input->dim_size() == 4,
             "reduce mean gpu only support 4-dim input");
  MACE_CHECK(axis_.size() == 2 && axis_[0] == 1 && axis_[1] == 2,
             "reduce mean gpu only support 1,2-axis reduce");
  index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);
  const index_t channel_blocks = RoundUpDiv4(channels);
  const uint32_t image_size = static_cast<uint32_t >(in_height * in_width);

  std::vector<uint32_t> gws(3);
  std::vector<uint32_t> lws(3);
  std::vector<index_t> output_shape{batch, 1, 1, channels};
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  auto runtime = context->device()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    const DataType dt = DataTypeToEnum<T>::value;
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("reduce_mean");
    built_options.emplace("-Dreduce_mean=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    if (runtime->gpu_type() != GPUType::QUALCOMM_ADRENO) {
      built_options.emplace("-DNON_QUALCOMM_ADRENO");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("reduce_mean",
                                              kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  if (runtime->gpu_type() == GPUType::QUALCOMM_ADRENO) {
    const uint32_t wave_size =
        static_cast<uint32_t>(runtime->GetKernelWaveSize(kernel_));
    gws = {4, (wave_size / 4), static_cast<uint32_t>(batch * channel_blocks)};
  } else {
    gws = {4, 16, static_cast<uint32_t>(batch * channel_blocks)};
  }
  lws = {gws[0], gws[1], 1};
  const int group_size = lws[0] * lws[1] * lws[2];
  const int partial_len = (image_size + group_size - 1) / group_size;
  const int remain_index = image_size % group_size;
  const float img_size_reciprocal = 1.f / (in_width * in_height);

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, (group_size * 4 * sizeof(T)),
                   nullptr);
    kernel_.setArg(idx++, static_cast<int32_t>(group_size));
    kernel_.setArg(idx++, static_cast<int32_t>(partial_len));
    kernel_.setArg(idx++, static_cast<int32_t>(remain_index));
    kernel_.setArg(idx++, static_cast<int32_t>(batch));
    kernel_.setArg(idx++, static_cast<int32_t>(in_height));
    kernel_.setArg(idx++, static_cast<int32_t>(in_width));
    kernel_.setArg(idx++, img_size_reciprocal);
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blocks));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      roundup_gws[i] = RoundUp(gws[i], lws[i]);
    }
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange,
        cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
        cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
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

#endif  // MACE_OPS_OPENCL_IMAGE_REDUCE_MEAN_H_

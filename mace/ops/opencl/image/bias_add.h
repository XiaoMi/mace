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
#ifndef MACE_OPS_OPENCL_IMAGE_BIAS_ADD_H_
#define MACE_OPS_OPENCL_IMAGE_BIAS_ADD_H_

#include "mace/ops/opencl/bias_add.h"

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
class BiasAddKernel : public OpenCLBiasAddKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *bias,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus BiasAddKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *bias,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("bias_add");
    built_options.emplace("-Dbias_add=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("bias_add", kernel_name,
                                              built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(bias->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);

  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
      if (lws[i] != 0) roundup_gws[i] = RoundUp(gws[i], lws[i]);
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

#endif  // MACE_OPS_OPENCL_IMAGE_BIAS_ADD_H_

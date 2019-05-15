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
#ifndef MACE_OPS_OPENCL_IMAGE_SPACE_TO_DEPTH_H_
#define MACE_OPS_OPENCL_IMAGE_SPACE_TO_DEPTH_H_

#include "mace/ops/opencl/space_to_depth.h"

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
class SpaceToDepthKernel : public OpenCLSpaceToDepthKernel {
 public:
  explicit SpaceToDepthKernel(const int block_size)
      : block_size_(block_size) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  const int block_size_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus SpaceToDepthKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_depth = input->dim(3);

  MACE_CHECK(input_depth < 4 || (input_depth % 4) == 0,
             "input channel should be dividable by 4");
  MACE_CHECK(
      (input_width % block_size_ == 0) && (input_height % block_size_ == 0),
      "input width and height should be dividable by block_size");

  const index_t output_height = input_height / block_size_;
  const index_t output_width = input_width / block_size_;
  const index_t output_depth = input_depth * block_size_ * block_size_;

  const index_t output_depth_blocks = RoundUpDiv4(output_depth);

  std::vector<index_t> output_shape = {batch, output_height, output_width,
                                       output_depth};

  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    const char *kernel_name = "space_to_depth";
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    if (input_depth < 4) {
      built_options.emplace(MakeString("-DDEPTH", input_depth));
    }
    built_options.emplace(kernel_name_ss.str());
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("space_to_depth",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(output_depth_blocks),
                           static_cast<uint32_t>(output_width),
                           static_cast<uint32_t>(output_height * batch)};
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(input_height));
    kernel_.setArg(idx++, static_cast<int32_t>(input_width));
    kernel_.setArg(idx++, static_cast<int32_t>(input_depth));
    kernel_.setArg(idx++, static_cast<int32_t>(block_size_));
    kernel_.setArg(idx++, static_cast<int32_t>(output_height));
    kernel_.setArg(idx++, static_cast<int32_t>(output_width));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key = Concat("space_to_depth", input->dim(0),
                                  input->dim(1), input->dim(2), input->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_SPACE_TO_DEPTH_H_

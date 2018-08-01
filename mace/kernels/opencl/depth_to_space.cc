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

#include "mace/kernels/depth_to_space.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus DepthToSpaceOpFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input, Tensor *output, StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_depth = input->dim(3);

  const char *kernel_name = nullptr;

  uint32_t gws[3];
  std::string tuning_key;
  index_t output_height, output_width, output_depth;
  if (d2s_) {
    output_height = input_height * block_size_;
    output_width = input_width * block_size_;
    output_depth = input_depth / (block_size_ * block_size_);
    MACE_CHECK(output_depth % 4 == 0, "output channel not support:")
        << output_depth;
    kernel_name = "depth_to_space";

    gws[0] = static_cast<uint32_t>(RoundUpDiv4(output_depth));
    gws[1] = static_cast<uint32_t>(output_width);
    gws[2] = static_cast<uint32_t>(output_height * batch);
    tuning_key = Concat("depth_to_space_opencl_kernel", batch, output_height,
                        output_width, output_depth);
  } else {
    output_height = input_height / block_size_;
    output_width = input_width / block_size_;
    output_depth = input_depth * block_size_ * block_size_;
    MACE_CHECK(input_depth % 4 == 0, "input channel not support:")
        << input_depth;
    kernel_name = "space_to_depth";

    gws[0] = static_cast<uint32_t>(RoundUpDiv4(input_depth));
    gws[1] = static_cast<uint32_t>(input_width);
    gws[2] = static_cast<uint32_t>(input_height * batch);
    tuning_key = Concat("space_to_depth_opencl_kernel", input->dim(0),
                        input->dim(1), input->dim(2), input->dim(3));
  }
  const index_t input_depth_blocks = RoundUpDiv4(input_depth);
  const index_t output_depth_blocks = RoundUpDiv4(output_depth);

  std::vector<index_t> output_shape = {batch, output_height, output_width,
                                       output_depth};

  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("depth_to_space",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    if (d2s_) {
      kernel_.setArg(idx++, static_cast<int32_t>(block_size_));
      kernel_.setArg(idx++, static_cast<int32_t>(input_height * batch));
      kernel_.setArg(idx++, static_cast<int32_t>(input_width));
      kernel_.setArg(idx++, static_cast<int32_t>(input_depth_blocks));
      kernel_.setArg(idx++, static_cast<int32_t>(output_width));
      kernel_.setArg(idx++, static_cast<int32_t>(output_depth_blocks));
    } else {
      kernel_.setArg(idx++, static_cast<int32_t>(block_size_));
      kernel_.setArg(idx++, static_cast<int32_t>(input_width));
      kernel_.setArg(idx++, static_cast<int32_t>(input_depth_blocks));
      kernel_.setArg(idx++, static_cast<int32_t>(output_height * batch));
      kernel_.setArg(idx++, static_cast<int32_t>(output_width));
      kernel_.setArg(idx++, static_cast<int32_t>(output_depth_blocks));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(gws, kwg_size_);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct DepthToSpaceOpFunctor<DeviceType::GPU, float>;
template struct DepthToSpaceOpFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

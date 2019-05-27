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

#include "mace/ops/opencl/image/pooling.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus PoolingKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const PoolingType pooling_type,
    const int *kernels,
    const int *strides,
    const Padding &padding_type,
    const std::vector<int> &padding_data,
    const int *dilations,
    const RoundType round_type,
    Tensor *output) {
  MACE_CHECK(dilations[0] == 1 && dilations[1] == 1)
    << "Pooling opencl kernel not support dilation yet";

  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {input->dim(3), input->dim(3),
                                       kernels[0], kernels[1]};

  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter_shape.data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), filter_shape.data(),
                   padding_data.data(), dilations, strides, round_type,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);

    if (pooling_type == MAX && input->dtype() == output->dtype()) {
      auto data_dt = input->dtype();
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(data_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(data_dt));
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    }
    if (pooling_type == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("pooling",
                                              kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {
      static_cast<uint32_t>(RoundUpDiv4(output->dim(3))),
      static_cast<uint32_t>(output->dim(2)),
      static_cast<uint32_t>(output->dim(0) * output->dim(1)),
  };
  MACE_OUT_OF_RANGE_INIT(kernel_);

  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(output->dim(1)));
    kernel_.setArg(idx++, paddings[0] / 2);
    kernel_.setArg(idx++, paddings[1] / 2);
    kernel_.setArg(idx++, strides[0]);
    kernel_.setArg(idx++, strides[1]);
    kernel_.setArg(idx++, kernels[0]);
    kernel_.setArg(idx++, kernels[1]);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = pooling::LocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("pooling_opencl_kernel_", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

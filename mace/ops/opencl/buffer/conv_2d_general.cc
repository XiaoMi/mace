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

#include "mace/core/op_context.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {
namespace conv2d {

MaceStatus Conv2dGeneral(OpContext *context,
                         cl::Kernel *kernel,
                         const Tensor *padded_input,
                         const Tensor *filter,
                         const Tensor *bias,
                         const int *strides,
                         const int *dilations,
                         const DataType dt,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float leakyrelu_coefficient,
                         const bool input_changed,
                         Tensor *output,
                         StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const index_t in_height = padded_input->dim(1);
  const index_t in_width = padded_input->dim(2);
  const index_t in_channel = padded_input->dim(3);

  const index_t filter_height = filter->dim(2);
  const index_t filter_width = filter->dim(3);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("conv2d");
    built_options.emplace("-Dconv2d=" + kernel_name);
    built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(padded_input->dtype()));
    built_options.emplace("-DOUT_DATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
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

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("conv_2d_buffer",
                                              kernel_name,
                                              built_options, kernel));
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(
                               RoundUpDiv4(channel) * RoundUpDiv4(width)),
                           static_cast<uint32_t>(height * batch)};

  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    auto filter_buffer_shape = filter->buffer_shape();
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->size());
    MACE_SET_2D_GWS_ARGS(*kernel, gws)
    kernel->setArg(idx++, *(padded_input->opencl_buffer()));
    kernel->setArg(idx++, *(filter->opencl_buffer()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_buffer()));
    }
    kernel->setArg(idx++, static_cast<int32_t>(in_height));
    kernel->setArg(idx++, static_cast<int32_t>(in_width));
    kernel->setArg(idx++, static_cast<int32_t>(padded_input->dim(3)));
    kernel->setArg(idx++, static_cast<int32_t>(filter_height));
    kernel->setArg(idx++, static_cast<int32_t>(filter_width));
    kernel->setArg(idx++,
                   static_cast<int32_t>(filter_buffer_shape[3]));
    kernel->setArg(idx++, static_cast<int32_t>(
        filter_buffer_shape[2] * filter_buffer_shape[3]
            * filter_buffer_shape[4]));
    kernel->setArg(idx++, static_cast<int32_t>(height));
    kernel->setArg(idx++, static_cast<int32_t>(width));
    kernel->setArg(idx++, static_cast<int32_t>(channel));
    kernel->setArg(idx++, strides[0]);
    kernel->setArg(idx++, strides[1]);
    kernel->setArg(idx++, static_cast<int32_t>(
        dilations[0] * in_width * in_channel));
    kernel->setArg(idx++, static_cast<int32_t>(
        dilations[1] * in_channel));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
    kernel->setArg(idx++, *(output->opencl_buffer()));
  }

  std::string tuning_key =
      Concat("conv2d_general_buffer", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3), filter_height, filter_width);
  std::vector<uint32_t> lws = {16, 4, 0};
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, *kernel, tuning_key, gws,
                                           lws, future));
  MACE_OUT_OF_RANGE_VALIDATION
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace conv2d
}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

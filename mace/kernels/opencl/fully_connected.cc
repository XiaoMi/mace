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

#include "mace/kernels/fully_connected.h"

#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

namespace {
template <typename T>
MaceStatus FCWXKernel(cl::Kernel *kernel,
                      const Tensor *input,
                      const Tensor *weight,
                      const Tensor *bias,
                      std::vector<index_t> *prev_input_shape,
                      Tensor *output,
                      const ActivationType activation,
                      std::vector<uint32_t> *gws,
                      std::vector<uint32_t> *lws,
                      const float relux_max_limit,
                      StatsFuture *future,
                      std::unique_ptr<BufferBase> *kernel_error) {
  MACE_CHECK_NOTNULL(gws);
  MACE_CHECK_NOTNULL(lws);
  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    const index_t batch = output->dim(0);
    const index_t output_size = output->dim(3);
    const index_t output_blocks = RoundUpDiv4(output_size);

    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(*kernel_error);
    NON_UNIFORM_WG_CONFIG;
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
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }
    if (runtime->gpu_type() != GPUType::QUALCOMM_ADRENO) {
      built_options.emplace("-DNON_QUALCOMM_ADRENO");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("fully_connected", kernel_name,
                                              built_options, kernel));

    if (runtime->gpu_type() == GPUType::QUALCOMM_ADRENO) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
      const uint32_t wave_size =
          static_cast<uint32_t>(runtime->GetKernelWaveSize(*kernel));

      *gws = {4, (wave_size / 4), static_cast<uint32_t>(batch * output_blocks)};

      const uint32_t kwg_size =
          static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
      const uint32_t inter_local_blks = kwg_size / ((*gws)[0] * (*gws)[1]);
      *lws = {(*gws)[0], (*gws)[1], inter_local_blks};
    } else {
      *gws = {4, 8, static_cast<uint32_t>(batch * output_blocks)};

      const uint32_t kwg_size =
          static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
      const uint32_t inter_local_blks = kwg_size / ((*gws)[0] * (*gws)[1]);
      *lws = {(*gws)[0], (*gws)[1], inter_local_blks};
    }
  }
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    const index_t batch = output->dim(0);
    const index_t output_blocks = RoundUpDiv4(output->dim(3));
    (*gws)[2] = static_cast<uint32_t>(batch * output_blocks);

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG_PTR;
    SET_3D_GWS_ARGS_PTR(kernel, *gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(weight->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, ((*lws)[0] * (*lws)[1] * (*lws)[2] * sizeof(float)),
                   nullptr);
    kernel->setArg(idx++, static_cast<int>(input->dim(1)));
    kernel->setArg(idx++, static_cast<int>(input->dim(2)));
    kernel->setArg(idx++, static_cast<int>(RoundUpDiv4(input->dim(3))));
    kernel->setArg(idx++, static_cast<int>(output_blocks));
    kernel->setArg(idx++, relux_max_limit);

    *prev_input_shape = input->shape();
  }
  cl::Event event;
  cl_int error;
  if (runtime->IsNonUniformWorkgroupsSupported()) {
    error = runtime->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange, cl::NDRange((*gws)[0], (*gws)[1], (*gws)[2]),
        cl::NDRange((*lws)[0], (*lws)[1], (*lws)[2]), nullptr, &event);
  } else {
    std::vector<uint32_t> roundup_gws(lws->size());
    for (size_t i = 0; i < lws->size(); ++i) {
      roundup_gws[i] = RoundUp((*gws)[i], (*lws)[i]);
    }
    error = runtime->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange,
        cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
        cl::NDRange((*lws)[0], (*lws)[1], (*lws)[2]), nullptr, &event);
  }
  OUT_OF_RANGE_VALIDATION(*kernel_error);
  MACE_CL_RET_STATUS(error);

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

template <typename T>
MaceStatus FCWTXKernel(cl::Kernel *kernel,
                       const Tensor *input,
                       const Tensor *weight,
                       const Tensor *bias,
                       std::vector<index_t> *prev_input_shape,
                       Tensor *output,
                       const ActivationType activation,
                       std::vector<uint32_t> *gws,
                       std::vector<uint32_t> *lws,
                       const float relux_max_limit,
                       StatsFuture *future,
                       std::unique_ptr<BufferBase> *kernel_error) {
  MACE_CHECK_NOTNULL(gws);
  MACE_CHECK_NOTNULL(lws);
  auto runtime = OpenCLRuntime::Global();
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(*kernel_error);
    NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected");
    built_options.emplace("-Dfully_connected=" + kernel_name);
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
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }
    MACE_RETURN_IF_ERROR(
        runtime->BuildKernel("fully_connected", kernel_name,
                             built_options, kernel));

    uint32_t kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
    *lws = {16, kwg_size / 16, 0};
  }
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    const index_t batch = output->dim(0);
    const index_t output_blocks = RoundUpDiv4(output->dim(3));

    *gws = {
        static_cast<uint32_t>(batch), static_cast<uint32_t>(output_blocks),
    };

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG_PTR;
    SET_2D_GWS_ARGS_PTR(kernel, *gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(weight->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, static_cast<int>(input->dim(1)));
    kernel->setArg(idx++, static_cast<int>(input->dim(2)));
    kernel->setArg(idx++, static_cast<int>(input->dim(3)));
    // FIXME handle flexable data type: half not supported
    kernel->setArg(idx++, relux_max_limit);

    *prev_input_shape = input->shape();
  }

  std::string tuning_key =
      Concat("fc_opencl_kernel", output->dim(0), output->dim(1), output->dim(2),
             output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(*kernel, tuning_key,
                                           gws->data(), *lws, future));

  OUT_OF_RANGE_VALIDATION(*kernel_error);
  return MACE_SUCCESS;
}
}  // namespace

template <typename T>
MaceStatus FullyConnectedFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  return FCWXKernel<T>(&kernel_, input, weight, bias, &input_shape_, output,
                       activation_, &gws_, &lws_, relux_max_limit_, future,
                       &kernel_error_);
}

template struct FullyConnectedFunctor<DeviceType::GPU, float>;

template struct FullyConnectedFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/fully_connected.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void FCWXKernel(cl::Kernel *kernel,
                const Tensor *input,
                const Tensor *weight,
                const Tensor *bias,
                std::vector<index_t> *prev_input_shape,
                Tensor *output,
                const ActivationType activation,
                std::vector<uint32_t> &gws,
                std::vector<uint32_t> &lws,
                const float relux_max_limit,
                StatsFuture *future) {
  MACE_CHECK(input->dim(3) % 4 == 0)
    << "FC width kernel only support input with 4x channel.";
  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected");
    kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected_width");
    built_options.emplace("-Dfully_connected_width=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
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

    *kernel =
        runtime->BuildKernel("fully_connected", kernel_name, built_options);

    const index_t batch = output->dim(0);
    const index_t output_size = output->dim(3);
    const index_t output_blocks = RoundUpDiv4(output_size);
    const uint32_t wave_size = runtime->GetKernelWaveSize(*kernel);

    gws = {4, (wave_size / 4), static_cast<uint32_t>(batch * output_blocks)};

    const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(*kernel);
    const uint32_t inter_local_blks = kwg_size / (gws[0] * gws[1]);
    lws = {gws[0], gws[1], inter_local_blks};

  }
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    const index_t batch = output->dim(0);
    const index_t output_blocks = RoundUpDiv4(output->dim(3));

    uint32_t idx = 0;
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(weight->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, (lws[0] * lws[1] * lws[2] * sizeof(float)), nullptr);
    kernel->setArg(idx++, static_cast<int>(input->dim(1)));
    kernel->setArg(idx++, static_cast<int>(input->dim(2)));
    kernel->setArg(idx++, static_cast<int>(RoundUpDiv4(input->dim(3))));
    kernel->setArg(idx++, static_cast<int>(output_blocks));
    kernel->setArg(idx++, relux_max_limit);

    gws[2] = static_cast<uint32_t>(batch * output_blocks);

    *prev_input_shape = input->shape();
  }
  cl::Event event;
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      *kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
  MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;

  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }

}

template <typename T>
void FCWTXKernel(cl::Kernel *kernel,
                 const Tensor *input,
                 const Tensor *weight,
                 const Tensor *bias,
                 std::vector<index_t> *prev_input_shape,
                 Tensor *output,
                 const ActivationType activation,
                 std::vector<uint32_t> &gws,
                 std::vector<uint32_t> &lws,
                 const float relux_max_limit,
                 StatsFuture *future) {
  if (kernel->get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("fully_connected");
    built_options.emplace("-Dfully_connected=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
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
    *kernel =
        runtime->BuildKernel("fully_connected", kernel_name, built_options);

    lws = {16, 64, 1};
  }
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
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

    const index_t batch = output->dim(0);
    const index_t output_blocks = RoundUpDiv4(output->dim(3));

    gws = {
        static_cast<uint32_t>(batch), static_cast<uint32_t>(output_blocks),
    };

    *prev_input_shape = input->shape();
  }

  std::stringstream ss;
  ss << "fc_opencl_kernel_" << output->dim(0) << "_" << output->dim(1) << "_"
     << output->dim(2) << "_" << output->dim(3);
  TuningOrRun2DKernel(*kernel, ss.str(), gws.data(), lws, future);

}

template <typename T>
void FullyConnectedFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  if (weight_type_ == BufferType::WEIGHT_HEIGHT) {
    FCWTXKernel<T>(&kernel_, input, weight, bias, &input_shape_, output,
                   activation_, gws_, lws_, relux_max_limit_, future);
  } else {
    FCWXKernel<T>(&kernel_, input, weight, bias, &input_shape_, output,
                  activation_, gws_, lws_, relux_max_limit_, future);
  }
};

template struct FullyConnectedFunctor<DeviceType::OPENCL, float>;

template struct FullyConnectedFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

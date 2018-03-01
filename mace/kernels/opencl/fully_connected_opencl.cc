//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/fully_connected.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template<typename T>
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

  const index_t batch = output->dim(0);
  const index_t output_size = output->dim(3);

  const index_t output_blocks = RoundUpDiv4(output_size);

  if (kernel_.get() == nullptr) {
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
    switch (activation_) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case PRELU:
        built_options.emplace("-DUSE_PRELU");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    kernel_ = runtime->BuildKernel("fully_connected", kernel_name, built_options);

    uint32_t idx = 0;
    kernel_.setArg(idx++,
                   *(static_cast<const cl::Image2D *>(input->buffer())));
    kernel_.setArg(idx++,
                   *(static_cast<const cl::Image2D *>(weight->buffer())));
    if (bias != nullptr) {
      kernel_.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(bias->buffer())));
    }
    kernel_.setArg(idx++,
                   *(static_cast<const cl::Image2D *>(output->buffer())));
    kernel_.setArg(idx++, static_cast<int>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int>(input->dim(3)));
    // FIXME handle flexable data type: half not supported
    kernel_.setArg(idx++, relux_max_limit_);
    kernel_.setArg(idx++, prelu_alpha_);
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(output_blocks),
  };
  const std::vector<uint32_t> lws = {16, 64, 1};
  std::stringstream ss;
  ss << "fc_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);

};

template
struct FullyConnectedFunctor<DeviceType::OPENCL, float>;

template
struct FullyConnectedFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

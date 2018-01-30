//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

extern void Conv2dOpencl(cl::Kernel *kernel,
                         const Tensor *input,
                         const Tensor *filter,
                         const Tensor *bias,
                         const int stride,
                         const int *padding,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float prelu_alpha,
                         const DataType dt,
                         Tensor *output,
                         StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("conv_2d");
    built_options.emplace("-Dconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    built_options.emplace("-DSTRIDE=" + ToString(stride));
    switch (activation) {
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
      defeult:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    auto runtime = OpenCLRuntime::Global();
    *kernel =
        runtime->BuildKernel("conv_2d", kernel_name, built_options);

    uint32_t idx = 0;
    kernel->setArg(idx++,
                          *(static_cast<const cl::Image2D *>(input->buffer())));
    kernel->setArg(idx++,
                          *(static_cast<const cl::Image2D *>(filter->buffer())));
    if (bias != nullptr) {
      kernel->setArg(idx++,
                            *(static_cast<const cl::Image2D *>(bias->buffer())));
    }
    kernel->setArg(idx++,
                          *(static_cast<const cl::Image2D *>(output->buffer())));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, prelu_alpha);
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<uint32_t>(height));
    kernel->setArg(idx++, static_cast<uint32_t>(width));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(0)));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(stride));
    kernel->setArg(idx++, padding[0] / 2);
    kernel->setArg(idx++, padding[1] / 2);
    kernel->setArg(idx++, dilations[0]);
    kernel->setArg(idx++, dilations[1]);
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::string tuning_key =
      Concat("conv2d_general_opencl_kernel_", activation, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  TuningOrRun3DKernel(*kernel, tuning_key, gws, lws, future);
}

}  // namespace kernels
}  // namespace mace

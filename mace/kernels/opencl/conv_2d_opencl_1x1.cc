//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

extern void Conv2dOpenclK1x1(cl::Kernel *kernel,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             const int stride,
                             const int *padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const DataType dt,
                             std::vector<index_t> *prev_input_shape,
                             Tensor *output,
                             StatsFuture *future,
                             bool *is_non_uniform_work_groups_supported,
                             uint32_t *kwg_size) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_blocks = RoundUpDiv4(width);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    *is_non_uniform_work_groups_supported =
        runtime->IsNonUniformWorkgroupsSupported();
    MACE_CHECK(input_batch == batch);

    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("conv_2d_1x1");
    built_options.emplace("-Dconv_2d_1x1=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    if (*is_non_uniform_work_groups_supported) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }
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

    *kernel = runtime->BuildKernel("conv_2d_1x1", kernel_name, built_options);
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    // FIXME handle flexable data type: half not supported
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, static_cast<int>(input_height));
    kernel->setArg(idx++, static_cast<int>(input_width));
    kernel->setArg(idx++, static_cast<int>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<int>(height));
    kernel->setArg(idx++, static_cast<int>(width));
    kernel->setArg(idx++, stride);
    kernel->setArg(idx++, gws[0]);
    kernel->setArg(idx++, gws[1]);
    kernel->setArg(idx++, gws[2]);

    *prev_input_shape = input->shape();

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const std::vector<uint32_t> lws = {8, *kwg_size / 64, 8, 1};
  std::string tuning_key =
      Concat("conv2d_1x1_opencl_kernel_", activation, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  TuningOrRun3DKernel(*kernel, tuning_key, gws, lws, future);
}

}  // namespace kernels
}  // namespace mace

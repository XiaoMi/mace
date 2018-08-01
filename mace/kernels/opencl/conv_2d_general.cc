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

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {
namespace {
// (inputs + weights + outputs) * array_size * sizeof(float)
const uint32_t kernel_cache_size = (4 + 4 + 4) * 4 * 4;
// TODO(liuqi): Fix the specific value.
const uint32_t lws_limit = 20;
std::vector<uint32_t> LocalWS(const uint32_t *gws,
                              const uint32_t kernel_size,
                              const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = OpenCLRuntime::Global()->device_global_mem_cache_size();
    uint32_t compute_units = OpenCLRuntime::Global()->device_compute_units();
    const uint32_t base =
        std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] = gws[0] / 4;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>((cache_size / kernel_cache_size / kernel_size /
                                    lws_size / compute_units) *
                                    8,
                                gws[2]);
    if (lws[2] == 0) {
      if (gws[2] < lws_limit) {
        lws[2] = gws[2];
      } else {
        lws[2] = base;
      }
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
  return lws;
}

}  // namespace

extern MaceStatus Conv2dOpencl(cl::Kernel *kernel,
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
                               uint32_t *kwg_size,
                               std::unique_ptr<BufferBase> *kernel_error) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(*kernel_error);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("conv_2d");
    built_options.emplace("-Dconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
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
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("conv_2d", kernel_name,
                                              built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  // Support different input size
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG_PTR;
    SET_3D_GWS_ARGS_PTR(kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<uint32_t>(height));
    kernel->setArg(idx++, static_cast<uint32_t>(width));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(3)));
    kernel->setArg(idx++, static_cast<uint32_t>(stride));
    kernel->setArg(idx++, padding[0] / 2);
    kernel->setArg(idx++, padding[1] / 2);
    kernel->setArg(idx++, dilations[0]);
    kernel->setArg(idx++, dilations[1]);

    *prev_input_shape = input->shape();
  }

  std::string tuning_key =
      Concat("conv2d_general_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3), filter->dim(2), filter->dim(3));
  std::vector<uint32_t> lws =
      LocalWS(gws, filter->dim(2) * filter->dim(3), *kwg_size);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(*kernel, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(*kernel_error);
  return MACE_SUCCESS;
}

}  // namespace kernels
}  // namespace mace

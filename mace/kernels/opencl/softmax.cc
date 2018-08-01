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

#include "mace/kernels/softmax.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

namespace {

std::vector<uint32_t> LocalWS(const uint32_t *gws, const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = OpenCLRuntime::Global()->device_global_mem_cache_size();
    uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    if (gws[0] < base) {
      lws[0] = gws[0];
    } else {
      lws[0] = gws[0] / base;
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(
        gws[2], kwg_size / (lws[0] * lws[1])), 1);
  }
  return lws;
}

}  // namespace

template <typename T>
MaceStatus SoftmaxFunctor<DeviceType::GPU, T>::operator()(const Tensor *logits,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  index_t batch = 0;
  index_t height = 0;
  index_t width = 0;
  index_t channels = 0;

  if (logits->dim_size() == 2) {
    batch = logits->dim(0);
    height = 1;
    width = 1;
    channels = logits->dim(1);

  } else if (logits->dim_size() == 4) {
    batch = logits->dim(0);
    height = logits->dim(1);
    width = logits->dim(2);
    channels = logits->dim(3);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  const index_t channel_blocks = RoundUpDiv4(channels);
  const int remain_channels = channel_blocks * 4 - channels;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("softmax");
    built_options.emplace("-Dsoftmax=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("softmax", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  if (!IsVecEqual(input_shape_, logits->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(logits->opencl_image()));
    kernel_.setArg(idx++, static_cast<int>(channels));
    kernel_.setArg(idx++, remain_channels);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = logits->shape();
  }

  std::vector<uint32_t> lws = LocalWS(gws, kwg_size_);
  std::string tuning_key =
      Concat("softmax_opencl_kernel", batch, height, width, channels);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct SoftmaxFunctor<DeviceType::GPU, float>;
template struct SoftmaxFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace

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

#include "mace/kernels/pooling.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

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
    lws[2] =
        std::min<uint32_t>(std::min<uint32_t>(gws[2], base), kwg_size / lws[1]);
    const uint32_t lws_size = lws[1] * lws[2];
    lws[0] = gws[0] / 4;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
    lws[0] = std::max<uint32_t>(std::min<uint32_t>(lws[0], kwg_size / lws_size),
                                1);
  }
  return lws;
}

}  // namespace

template <typename T>
MaceStatus PoolingFunctor<DeviceType::GPU, T>::operator()(const Tensor *input,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1)
      << "Pooling opencl kernel not support dilation yet";

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    const DataType dt = DataTypeToEnum<T>::value;
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);

    if (pooling_type_ == MAX && input->dtype() == output->dtype()) {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
      built_options.emplace(dt == DT_HALF ? "-DFP16" : "");
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    }
    if (pooling_type_ == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("pooling",
                                              kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  std::vector<uint32_t> gws;
  if (!IsVecEqual(input_shape_, input->shape())) {
    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {input->dim(3), input->dim(3),
                                         kernels_[0], kernels_[1]};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      kernels::CalcNHWCPaddingAndOutputSize(
          input->shape().data(), filter_shape.data(), dilations_, strides_,
          padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input->shape().data(), filter_shape.data(),
                     paddings_.data(), dilations_, strides_, RoundType::CEIL,
                     output_shape.data());
    }

    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                    &output_image_shape);
    MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

    index_t batch = output->dim(0);
    index_t out_height = output->dim(1);
    index_t out_width = output->dim(2);
    index_t channels = output->dim(3);

    index_t channel_blocks = (channels + 3) / 4;

    gws = {
        static_cast<uint32_t>(channel_blocks), static_cast<uint32_t>(out_width),
        static_cast<uint32_t>(batch * out_height),
    };

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(output->dim(1)));
    kernel_.setArg(idx++, paddings[0] / 2);
    kernel_.setArg(idx++, paddings[1] / 2);
    kernel_.setArg(idx++, strides_[0]);
    kernel_.setArg(idx++, strides_[1]);
    kernel_.setArg(idx++, kernels_[0]);
    kernel_.setArg(idx++, kernels_[1]);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  } else {
    index_t batch = output->dim(0);
    index_t out_height = output->dim(1);
    index_t out_width = output->dim(2);
    index_t channels = output->dim(3);

    index_t channel_blocks = (channels + 3) / 4;

    gws = {
        static_cast<uint32_t>(channel_blocks), static_cast<uint32_t>(out_width),
        static_cast<uint32_t>(batch * out_height),
    };
  }

  const std::vector<uint32_t> lws = LocalWS(gws.data(), kwg_size_);
  std::string tuning_key =
      Concat("pooling_opencl_kernel_", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws.data(), lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct PoolingFunctor<DeviceType::GPU, float>;
template struct PoolingFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace

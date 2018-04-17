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

template <typename T>
void PoolingFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                       Tensor *output,
                                                       StatsFuture *future) {
  MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1)
      << "Pooling opencl kernel not support dilation yet";

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    const DataType dt = DataTypeToEnum<T>::value;
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);

    if (pooling_type_ == MAX && input->dtype() == output->dtype()) {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
      built_options.emplace(dt == DT_HALF ? "-DFP16" : "");
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    }
    if (pooling_type_ == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    if (runtime->IsOutOfRangeCheckEnabled()) {
      built_options.emplace("-DOUT_OF_RANGE_CHECK");
      kernel_error_ = std::move(std::unique_ptr<Buffer>(
            new Buffer(GetDeviceAllocator(DeviceType::OPENCL), 1)));
      kernel_error_->Map(nullptr);
      *(kernel_error_->mutable_data<char>()) = 0;
      kernel_error_->UnMap();
    }
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
    kernel_ = runtime->BuildKernel("pooling", kernel_name, built_options);

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  std::vector<uint32_t> gws;
  if (!IsVecEqual(input_shape_, input->shape())) {
    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {kernels_[0], kernels_[1],
                                         input->dim(3), input->dim(3)};

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
    output->ResizeImage(output_shape, output_image_shape);

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
    if (runtime->IsOutOfRangeCheckEnabled()) {
      kernel_.setArg(idx++,
          *(static_cast<cl::Buffer *>(kernel_error_->buffer())));
    }
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      kernel_.setArg(idx++, gws[0]);
      kernel_.setArg(idx++, gws[1]);
      kernel_.setArg(idx++, gws[2]);
    }
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(output->dim(1)));
    kernel_.setArg(idx++, paddings[0] / 2);
    kernel_.setArg(idx++, paddings[1] / 2);
    kernel_.setArg(idx++, strides_[0]);
    kernel_.setArg(idx++, kernels_[0]);
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

  std::vector<uint32_t> lws = {8, kwg_size_ / 64, 8, 1};
  std::stringstream ss;
  ss << "pooling_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws.data(), lws, future);

  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_error_->Map(nullptr);
    char *kerror_code = kernel_error_->mutable_data<char>();
    MACE_CHECK(*kerror_code == 0) << "Kernel error code: " << *kerror_code;
    kernel_error_->UnMap();
  }
}

template struct PoolingFunctor<DeviceType::OPENCL, float>;
template struct PoolingFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

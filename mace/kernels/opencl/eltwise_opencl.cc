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

#include "mace/kernels/eltwise.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void EltwiseFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input0,
                                                       const Tensor *input1,
                                                       Tensor *output,
                                                       StatsFuture *future) {
  if (input1 != nullptr) {
    MACE_CHECK(input0->dim_size() == input1->dim_size())
      << "Inputs of Eltwise op must be same shape";
    if (input0->size() != input1->size()) {
      if (input0->size() < input1->size()) {
        std::swap(input0, input1);
      }
      MACE_CHECK(input0->dim(0) == input1->dim(0) &&
          input1->dim(1) == 1 &&
          input1->dim(2) == 1 &&
          input0->dim(3) == input1->dim(3))
        << "Element-Wise op only support channel dimension broadcast";
    }
  }
  output->ResizeLike(input0);
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t batch_height_pixels = batch * height;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(batch_height_pixels)};

  auto runtime = OpenCLRuntime::Global();
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (input1 == nullptr) {
      built_options.emplace(MakeString("-DINPUT_TYPE=1"));
    } else if (input0->size() != input1->size()) {
      built_options.emplace(MakeString("-DINPUT_TYPE=2"));
    }
    if (!coeff_.empty()) built_options.emplace("-DCOEFF_SUM");

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
    kernel_ = runtime->BuildKernel("eltwise", kernel_name, built_options);

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  if (!IsVecEqual(input_shape_, input0->shape())) {
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
    kernel_.setArg(idx++, *(input0->opencl_image()));
    if (input1 == nullptr) {
      kernel_.setArg(idx++, value_);
    } else {
      kernel_.setArg(idx++, *(input1->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    if (!coeff_.empty()) {
      kernel_.setArg(idx++, coeff_[0]);
      kernel_.setArg(idx++, coeff_[1]);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input0->shape();
  }

  const std::vector<uint32_t> lws = {8, kwg_size_ / 64, 8, 0};
  std::stringstream ss;
  ss << "eltwise_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_error_->Map(nullptr);
    char *kerror_code = kernel_error_->mutable_data<char>();
    MACE_CHECK(*kerror_code == 0) << "Kernel error code: " << *kerror_code;
    kernel_error_->UnMap();
  }
}

template struct EltwiseFunctor<DeviceType::OPENCL, float>;
template struct EltwiseFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

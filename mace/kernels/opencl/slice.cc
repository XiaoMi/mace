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

#include "mace/kernels/slice.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus SliceFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const std::vector<Tensor *> &output_list,
    StatsFuture *future) {
  const index_t input_channels = input->dim(3);
  const size_t outputs_count = output_list.size();
  const index_t output_channels = input_channels / outputs_count;
  MACE_CHECK(output_channels % 4 == 0)
      << "output channels of slice op must be divisible by 4";
  std::vector<index_t> output_shape(
      {input->dim(0), input->dim(1), input->dim(2), output_channels});

  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  for (size_t i = 0; i < outputs_count; ++i) {
    MACE_RETURN_IF_ERROR(
        output_list[i]->ResizeImage(output_shape, image_shape));
  }

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("slice");
    built_options.emplace("-Dslice=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(DataTypeToEnum<T>::value));
    if (runtime->IsOutOfRangeCheckEnabled()) {
      built_options.emplace("-DOUT_OF_RANGE_CHECK");
      kernel_error_ = std::move(std::unique_ptr<Buffer>(
          new Buffer(GetDeviceAllocator(DeviceType::GPU))));
      MACE_RETURN_IF_ERROR(kernel_error_->Allocate(1));
      kernel_error_->Map(nullptr);
      *(kernel_error_->mutable_data<char>()) = 0;
      kernel_error_->UnMap();
    }
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("slice",
                                              kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  const index_t channel_blk = RoundUpDiv4(output_channels);

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(input->dim(2)),
      static_cast<uint32_t>(input->dim(0) * input->dim(1)),
  };

  const std::vector<uint32_t> lws = Default3DLocalWS(gws, kwg_size_);
  cl::Event event;
  CallStats call_stats{INT64_MAX, 0};
  for (size_t i = 0; i < outputs_count; ++i) {
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
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blk * i));
    kernel_.setArg(idx++, *(output_list[i]->opencl_image()));

    cl_int error;
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    } else {
      std::vector<uint32_t> roundup_gws(lws.size());
      for (size_t j = 0; j < 3; ++j) {
        roundup_gws[j] = RoundUp(gws[j], lws[j]);
      }

      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange,
          cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    if (runtime->IsOutOfRangeCheckEnabled()) {
      kernel_error_->Map(nullptr);
      char *kerror_code = kernel_error_->mutable_data<char>();
      MACE_CHECK(*kerror_code == 0) << "Kernel error code: " << *kerror_code;
      kernel_error_->UnMap();
    }
    if (future != nullptr && runtime->is_profiling_enabled()) {
      event.wait();
      CallStats tmp_stats;
      runtime->GetCallStats(event, &tmp_stats);
      call_stats.start_micros =
          std::min<int64_t>(tmp_stats.start_micros, call_stats.start_micros);
      call_stats.end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
    }
  }
  if (future != nullptr) {
    future->wait_fn = [runtime, call_stats](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = call_stats.start_micros;
        stats->end_micros = stats->start_micros + call_stats.end_micros;
      }
    };
  }

  return MACE_SUCCESS;
}

template struct SliceFunctor<DeviceType::GPU, float>;
template struct SliceFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace

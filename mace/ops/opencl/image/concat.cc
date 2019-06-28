// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/opencl/image/concat.h"

#include <algorithm>
#include <set>
#include <string>

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace concat {
namespace {
std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                              const uint32_t *gws,
                              const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = runtime->device_global_mem_cache_size();
    uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] = std::min<uint32_t>(base, kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] =
        std::max<uint32_t>(std::min<uint32_t>(base, kwg_size / lws_size), 1);
  }
  return lws;
}

}  // namespace


MaceStatus Concat2(OpContext *context,
                   cl::Kernel *kernel,
                   const Tensor *input0,
                   const Tensor *input1,
                   std::vector<index_t> *prev_input_shape,
                   Tensor *output,
                   uint32_t *kwg_size) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const int channel_blk = RoundUpDiv4(channel);
  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(width),
      static_cast<uint32_t>(batch * height),
  };

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel");
    built_options.emplace("-Dconcat_channel=" + kernel_name);
    if (input0->dtype() == output->dtype()) {
      auto data_dt = input0->dtype();
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(data_dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(data_dt));
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    }

    if (input0->dim(3) % 4 == 0) {
      built_options.emplace("-DDIVISIBLE_FOUR");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("concat", kernel_name,
                                              built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }
  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (!IsVecEqual(*prev_input_shape, input0->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(*kernel);
    MACE_SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++,
                   *(static_cast<const cl::Image2D *>(input0->opencl_image())));
    kernel->setArg(idx++,
                   *(static_cast<const cl::Image2D *>(input1->opencl_image())));
    kernel->setArg(idx++, static_cast<int32_t>(input0->dim(3)));
    kernel->setArg(idx++, static_cast<int32_t>(input1->dim(3)));
    kernel->setArg(idx++,
                   *(static_cast<cl::Image2D *>(output->opencl_image())));

    *prev_input_shape = input0->shape();
  }

  const std::vector<uint32_t> lws = LocalWS(runtime, gws, *kwg_size);
  std::string tuning_key =
      Concat("concat_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ConcatN(OpContext *context,
                   cl::Kernel *kernel,
                   const std::vector<const Tensor *> &input_list,
                   Tensor *output,
                   uint32_t *kwg_size) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel_multi");
    built_options.emplace("-Dconcat_channel_multi=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("concat", kernel_name,
                                              built_options, kernel));
    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const int inputs_count = input_list.size();
  index_t chan_blk_offset = 0;
  cl::Event event;
  CallStats call_stats{INT64_MAX, 0};

  MACE_OUT_OF_RANGE_INIT(*kernel);
  for (int i = 0; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    index_t input_channel_blk = input->dim(3) / 4;
    const uint32_t gws[3] = {
        static_cast<uint32_t>(input_channel_blk), static_cast<uint32_t>(width),
        static_cast<uint32_t>(batch * height),
    };
    const std::vector<uint32_t> lws = LocalWS(runtime, gws, *kwg_size);

    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(*kernel);
    MACE_SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, static_cast<int32_t>(chan_blk_offset));
    kernel->setArg(idx++, *(output->opencl_image()));

    chan_blk_offset += input_channel_blk;
    cl_int error;
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      error = runtime->command_queue().enqueueNDRangeKernel(
          *kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    } else {
      std::vector<uint32_t> roundup_gws(lws.size());
      for (size_t j = 0; j < 3; ++j) {
        roundup_gws[j] = RoundUp(gws[j], lws[j]);
      }
      error = runtime->command_queue().enqueueNDRangeKernel(
          *kernel, cl::NullRange,
          cl::NDRange(roundup_gws[0], roundup_gws[1], roundup_gws[2]),
          cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    MACE_OUT_OF_RANGE_VALIDATION;
    if (context->future() != nullptr && runtime->is_profiling_enabled()) {
      event.wait();
      CallStats tmp_stats;
      runtime->GetCallStats(event, &tmp_stats);
      call_stats.start_micros =
          std::min<int64_t>(tmp_stats.start_micros, call_stats.start_micros);
      call_stats.end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
    }
  }
  if (context->future() != nullptr) {
    context->future()->wait_fn = [call_stats](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = call_stats.start_micros;
        stats->end_micros = stats->start_micros + call_stats.end_micros;
      }
    };
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace concat


MaceStatus ConcatKernel::Compute(
    OpContext *context,
    const std::vector<const Tensor *> &input_list,
    const int32_t axis,
    Tensor *output) {
  const int inputs_count = input_list.size();

  const Tensor *input0 = input_list[0];

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis] += input->dim(axis);
  }
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  switch (inputs_count) {
    case 2:
      return concat::Concat2(
          context, &kernel_, input_list[0], input_list[1],
          &input_shape_, output, &kwg_size_);
    default:
      return concat::ConcatN(context,
                             &kernel_,
                             input_list,
                             output,
                             &kwg_size_);
  }
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

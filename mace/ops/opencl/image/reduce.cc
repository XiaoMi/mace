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

#include "mace/ops/opencl/image/reduce.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

namespace {
const index_t TILE_SIZE = 16;

cl::Image *GetScratchImage(OpContext *context, MemoryType mem_type,
                           DataType dtype, const std::vector<index_t> &shape) {
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(shape, BufferContentType::IN_OUT_CHANNEL,
                              &image_shape);

  auto *runtime = context->runtime();
  MemInfo mem_info(mem_type, dtype, MemInfo::IndexT(image_shape));
  auto mace_image = runtime->ObtainBuffer(mem_info, RENT_SCRATCH);
  cl::Image *image = mace_image->mutable_memory<cl::Image>();

  return image;
}

}  // namespace

MaceStatus ReduceKernel::BuildReduceKernel(OpenclExecutor *executor,
                                           bool divisable_by_four) {
  std::set<std::string> built_options;
  MACE_OUT_OF_RANGE_CONFIG;
  MACE_NON_UNIFORM_WG_CONFIG;

  std::vector<int> hw_axis = {1, 2};
  std::vector<int> c_axis = {3};
  std::string kernel_name;
  if (hw_axis == axis_) {
    kernel_name = "reduce_hw";
  } else if (c_axis == axis_) {
    kernel_name = "reduce_c";
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
  if (!divisable_by_four) {
    built_options.emplace("-DNOT_DIVISIBLE_FOUR");
  }
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
  built_options.emplace(MakeString("-DREDUCE_TYPE=", reduce_type_));
  MACE_RETURN_IF_ERROR(executor->BuildKernel(
      "reduce", obfuscated_kernel_name, built_options, &kernel_));
  kwg_size_ =
      static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(kernel_));

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ReduceKernel::GraduallyComputeReduceHW(
    OpContext *context, const index_t batch, const index_t channel_blocks,
    const index_t in_height, const index_t in_width,
    const index_t out_height, const index_t out_width,
    const index_t org_height, const index_t org_width,
    const cl::Image *input, cl::Image *output,
    std::vector<StatsFuture> *futures) {
  MACE_OUT_OF_RANGE_DEFINITION;
  auto *executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  if (kernel_.get() == nullptr) {
    MACE_RETURN_IF_ERROR(BuildReduceKernel(executor));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(out_width),
                           static_cast<uint32_t>(out_height),
                           static_cast<uint32_t>(batch * channel_blocks)};
  std::vector<uint32_t> lws = Default3DLocalWS(executor, gws, kwg_size_);

  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
  MACE_SET_3D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *input);
  kernel_.setArg(idx++, static_cast<int>(out_height));
  kernel_.setArg(idx++, static_cast<int>(out_width));
  kernel_.setArg(idx++, static_cast<int>(in_height));
  kernel_.setArg(idx++, static_cast<int>(in_width));
  kernel_.setArg(idx++, static_cast<int>(org_height));
  kernel_.setArg(idx++, static_cast<int>(org_width));
  kernel_.setArg(idx++, static_cast<int>(channel_blocks));
  kernel_.setArg(idx++, *output);

  std::string tuning_key = Concat(
      "reduce_hw_opencl_kernel", gws[0], gws[1], gws[2]);

  futures->push_back(StatsFuture());
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, kernel_, tuning_key,
                                           gws, lws, &(futures->back()), context));
  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus ReduceKernel::ReduceHW(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t org_height = input->dim(1);
  const index_t org_width = input->dim(2);
  index_t in_height = org_height;
  index_t in_width = org_width;
  const index_t channels = input->dim(3);
  const index_t channel_blocks = RoundUpDiv4(channels);

  std::vector<index_t> output_shape{batch, 1, 1, channels};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  MaceStatus result = MaceStatus::MACE_RUNTIME_ERROR;
  auto *input_image = input->memory<cl::Image>();
  auto *output_image = output->mutable_memory<cl::Image>();
  std::vector<StatsFuture> futures;
  if (in_height <= TILE_SIZE && in_width <= TILE_SIZE) {
    result = GraduallyComputeReduceHW(context, batch, channel_blocks, in_height,
                                      in_width, 1, 1, org_height, org_width,
                                      input_image, output_image,
                                      &futures);
  } else {
    auto out_height = RoundUpDiv(in_height, TILE_SIZE);
    auto out_width = RoundUpDiv(in_width, TILE_SIZE);
    const std::vector<index_t> inter_shape =
        {{batch, out_height, out_width, channels}};
    cl::Image *inter_image = GetScratchImage(context, input->memory_type(),
                                             input->dtype(), inter_shape);

    result = GraduallyComputeReduceHW(context, batch, channel_blocks, in_height,
                                      in_width, out_height, out_width,
                                      org_height, org_width,
                                      input_image, inter_image,
                                      &futures);
    MACE_RETURN_IF_ERROR(result);

    in_height = out_height;
    in_width = out_width;
    out_height = RoundUpDiv(in_height, TILE_SIZE);
    out_width = RoundUpDiv(in_width, TILE_SIZE);

    if (in_height > TILE_SIZE || in_width > TILE_SIZE) {
      const std::vector<index_t> inter2_shape =
          {{batch, out_height, out_width, channels}};
      cl::Image *inter2_image = GetScratchImage(context, input->memory_type(),
                                                input->dtype(), inter2_shape);

      while (out_height > 1 || out_width > 1) {
        result = GraduallyComputeReduceHW(context, batch, channel_blocks,
                                          in_height, in_width, out_height,
                                          out_width, org_height, org_width,
                                          inter_image, inter2_image,
                                          &futures);
        MACE_RETURN_IF_ERROR(result);
        in_height = out_height;
        in_width = out_width;
        out_height = RoundUpDiv(in_height, TILE_SIZE);
        out_width = RoundUpDiv(in_width, TILE_SIZE);
        std::swap(inter_image, inter2_image);
      }
    }

    result = GraduallyComputeReduceHW(context, batch, channel_blocks, in_height,
                                      in_width, 1, 1, org_height, org_width,
                                      inter_image, output_image,
                                      &futures);
  }

  MergeMultipleFutureWaitFn(futures, context->future());
  return result;
}

MaceStatus ReduceKernel::GraduallyComputeReduceC(
    OpContext *context,
    const index_t batch,
    const index_t height,
    const index_t width,
    const index_t channels,
    const index_t channel_blocks,
    const index_t out_ch_blks,
    const index_t in_ch_blks,
    const cl::Image *input, cl::Image *output,
    std::vector<StatsFuture> *futures) {
  MACE_OUT_OF_RANGE_DEFINITION;
  auto *executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  if (kernel_.get() == nullptr) {
    bool divisable_by_four = (0 == (channels & 0x3));
    MACE_RETURN_IF_ERROR(BuildReduceKernel(executor, divisable_by_four));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(out_ch_blks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  std::vector<uint32_t> lws = Default3DLocalWS(executor, gws, kwg_size_);

  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
  MACE_SET_3D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *input);
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<int>(width));
  kernel_.setArg(idx++, static_cast<int>(channels));
  kernel_.setArg(idx++, static_cast<int>(channel_blocks));
  kernel_.setArg(idx++, static_cast<int>(in_ch_blks));
  kernel_.setArg(idx++, *output);

  std::string tuning_key = Concat(
      "reduce_c_opencl_kernel", gws[0], gws[1], gws[2]);

  futures->push_back(StatsFuture());
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, kernel_, tuning_key,
                                           gws, lws, &(futures->back()), context));
  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}
MaceStatus ReduceKernel::ReduceC(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);
  const index_t channel_blocks = RoundUpDiv4(channels);
  index_t in_ch_blks = channel_blocks;
  std::vector<index_t> output_shape{batch, height, width, 1};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  MaceStatus result = MaceStatus::MACE_RUNTIME_ERROR;
  auto *input_image = input->memory<cl::Image>();
  auto *output_image = output->mutable_memory<cl::Image>();
  std::vector<StatsFuture> futures;
  futures.reserve(1 + channel_blocks / TILE_SIZE);
  if (in_ch_blks <= TILE_SIZE) {
    result = GraduallyComputeReduceC(context,
                                     batch,
                                     height,
                                     width,
                                     channels,
                                     channel_blocks,
                                     1,
                                     in_ch_blks,
                                     input_image, output_image,
                                     &futures);
    MACE_RETURN_IF_ERROR(result);
  } else {
    index_t out_ch_blks = RoundUpDiv(in_ch_blks, TILE_SIZE);
    const std::vector<index_t> inter_shape = {batch, height, width, 4 * out_ch_blks};
    cl::Image *inter_image = GetScratchImage(context, input->memory_type(),
                                             input->dtype(), inter_shape);
    result = GraduallyComputeReduceC(context,
                                     batch,
                                     height,
                                     width,
                                     channels,
                                     channel_blocks,
                                     out_ch_blks,
                                     in_ch_blks,
                                     input_image, inter_image,
                                     &futures);
    MACE_RETURN_IF_ERROR(result);
    in_ch_blks = out_ch_blks;
    if (in_ch_blks > TILE_SIZE) {
      out_ch_blks = RoundUpDiv(in_ch_blks, TILE_SIZE);
      const std::vector<index_t> inter2_shape = {batch, height, width, 4 * out_ch_blks};
      cl::Image *inter2_image = GetScratchImage(context, input->memory_type(),
                                                input->dtype(), inter2_shape);
      while (out_ch_blks > 1) {
        result = GraduallyComputeReduceC(context,
                                         batch,
                                         height,
                                         width,
                                         channels,
                                         channel_blocks,
                                         out_ch_blks,
                                         in_ch_blks,
                                         inter_image, inter2_image,
                                         &futures);
        MACE_RETURN_IF_ERROR(result);
        in_ch_blks = out_ch_blks;
        out_ch_blks = RoundUpDiv(in_ch_blks, TILE_SIZE);
        std::swap(inter_image, inter2_image);
      }
    }

    result = GraduallyComputeReduceC(context,
                                     batch,
                                     height,
                                     width,
                                     channels,
                                     channel_blocks,
                                     1,
                                     in_ch_blks,
                                     inter_image, output_image,
                                     &futures);
    MACE_RETURN_IF_ERROR(result);
  }

  MergeMultipleFutureWaitFn(futures, context->future());
  return result;
}

MaceStatus ReduceKernel::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  MACE_CHECK_NOTNULL(input);
  int input_dim = input->dim_size();
  int axis_size = axis_.size();
  for (int i = 0; i < axis_size; ++i) {
    axis_[i] = (axis_[i] < 0) ? (axis_[i] + input_dim) : axis_[i];
  }
  std::vector<int> hw_axis = {1, 2};
  std::vector<int> c_axis = {3};
  if (hw_axis == axis_) {
    MACE_RETURN_IF_ERROR(ReduceHW(context, input, output));
  } else if (c_axis == axis_) {
    MACE_RETURN_IF_ERROR(ReduceC(context, input, output));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

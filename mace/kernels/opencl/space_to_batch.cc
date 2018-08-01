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

#ifndef MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_
#define MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

#include "mace/kernels/space_to_batch.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus SpaceToBatchFunctor<DeviceType::GPU, T>::operator()(
    Tensor *space_tensor, Tensor *batch_tensor, StatsFuture *future) {
  std::vector<index_t> output_shape(4, 0);
  if (b2s_) {
    CalculateBatchToSpaceOutputShape(batch_tensor, DataFormat::NHWC,
                                     output_shape.data());
  } else {
    CalculateSpaceToBatchOutputShape(space_tensor, DataFormat::NHWC,
                                     output_shape.data());
  }

  const char *kernel_name = nullptr;
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  if (b2s_) {
    MACE_RETURN_IF_ERROR(
        space_tensor->ResizeImage(output_shape, output_image_shape));
    kernel_name = "batch_to_space";
  } else {
    MACE_RETURN_IF_ERROR(
        batch_tensor->ResizeImage(output_shape, output_image_shape));
    kernel_name = "space_to_batch";
  }
  const uint32_t chan_blk = RoundUpDiv4<uint32_t>(batch_tensor->dim(3));
  const uint32_t gws[3] = {
      chan_blk, static_cast<uint32_t>(batch_tensor->dim(2)),
      static_cast<uint32_t>(batch_tensor->dim(0) * batch_tensor->dim(1))};

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(DataTypeToEnum<T>::value));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("space_to_batch",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  if (!IsVecEqual(space_shape_, space_tensor->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    if (b2s_) {
      kernel_.setArg(idx++, *(batch_tensor->opencl_image()));
      kernel_.setArg(idx++, *(space_tensor->opencl_image()));
    } else {
      kernel_.setArg(idx++, *(space_tensor->opencl_image()));
      kernel_.setArg(idx++, *(batch_tensor->opencl_image()));
    }
    kernel_.setArg(idx++, block_shape_[0]);
    kernel_.setArg(idx++, block_shape_[1]);
    kernel_.setArg(idx++, paddings_[0]);
    kernel_.setArg(idx++, paddings_[2]);
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(0)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));

    space_shape_ = space_tensor->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(gws, kwg_size_);
  std::string tuning_key =
      Concat(kernel_name, batch_tensor->dim(0), batch_tensor->dim(1),
             batch_tensor->dim(2), batch_tensor->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct SpaceToBatchFunctor<DeviceType::GPU, float>;
template struct SpaceToBatchFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

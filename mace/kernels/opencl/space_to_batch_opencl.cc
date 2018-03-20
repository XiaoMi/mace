//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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
void SpaceToBatchFunctor<DeviceType::OPENCL, T>::operator()(
    Tensor *space_tensor,
    const std::vector<index_t> &output_shape,
    Tensor *batch_tensor,
    StatsFuture *future) {
  const char *kernel_name = nullptr;
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  if (b2s_) {
    space_tensor->ResizeImage(output_shape, output_image_shape);
    kernel_name = "batch_to_space";
  } else {
    batch_tensor->ResizeImage(output_shape, output_image_shape);
    kernel_name = "space_to_batch";
  }
  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(DataTypeToEnum<T>::value));
    kernel_ =
        runtime->BuildKernel("space_to_batch", kernel_name, built_options);
  }
  if (!IsVecEqual(space_shape_, space_tensor->shape())) {
    uint32_t idx = 0;
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
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));

    space_shape_ = space_tensor->shape();
  }

  const uint32_t chan_blk = RoundUpDiv4<uint32_t>(batch_tensor->dim(3));
  const uint32_t gws[3] = {
      chan_blk, static_cast<uint32_t>(batch_tensor->dim(2)),
      static_cast<uint32_t>(batch_tensor->dim(0) * batch_tensor->dim(1))};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::stringstream ss;
  ss << kernel_name << "_" << batch_tensor->dim(0) << "_"
     << batch_tensor->dim(1) << "_" << batch_tensor->dim(2) << "_"
     << batch_tensor->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct SpaceToBatchFunctor<DeviceType::OPENCL, float>;
template struct SpaceToBatchFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

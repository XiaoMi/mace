//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_
#define MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/space_to_batch.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

template <>
void SpaceToBatchFunctor<DeviceType::OPENCL, float>::operator()(Tensor *space_tensor,
                                                                const Tensor *block_shape_tensor,
                                                                const Tensor *paddings_tensor,
                                                                Tensor *batch_tensor) {
  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(space_tensor->dtype()));
  auto s2b_kernel = runtime->BuildKernel("space_to_batch", "space_to_batch", built_options);

  uint32_t idx = 0;
  s2b_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(space_tensor->buffer())));
  s2b_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(block_shape_tensor->buffer())));
  s2b_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(paddings_tensor->buffer())));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(0)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(3)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(3)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(b2s_));
  s2b_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(batch_tensor->buffer())));

  const uint32_t gws[3] = {static_cast<uint32_t>(batch_tensor->dim(0)),
                           static_cast<uint32_t>(batch_tensor->dim(1)),
                           static_cast<uint32_t>(batch_tensor->dim(2) * batch_tensor->dim(3))};
  const uint32_t lws[3] = {static_cast<uint32_t>(1),
                           static_cast<uint32_t>(8),
                           static_cast<uint32_t>(128)};
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      s2b_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS);
}

} //  namespace kernels
} //  namespace mace
#endif //  MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

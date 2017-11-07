//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_
#define MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <bool B2S = false>
void SpaceToBatch(Tensor *space_tensor,
                  const int block_height,
                  const int block_width,
                  Tensor *batch_tensor,
                  const std::vector<cl::Event> *waiting_events,
                  cl::Event *event) {
  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto s2b_kernel = cl::Kernel(program, "space_to_batch");

  uint32_t idx = 0;
  s2b_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(space_tensor->buffer())));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(0)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(3)));
  s2b_kernel.setArg(idx++, block_height);
  s2b_kernel.setArg(idx++, block_width);
  s2b_kernel.setArg(idx++, static_cast<int32_t>(B2S));
  s2b_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(batch_tensor->buffer())));

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
      waiting_events,
      event);
  MACE_CHECK(error == CL_SUCCESS);
}

} //  namespace kernels
} //  namespace mace
#endif //  MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

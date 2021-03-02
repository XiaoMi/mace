// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {
namespace runtimes {
namespace opencl {

MaceStatus BufferTypeTransform(
    OpContext *context,
    cl::Kernel *kernel,
    const Tensor *input,
    Tensor *output) {
  MACE_RETURN_IF_ERROR(output->ResizeLike(input));

  auto runtime = input->GetCurRuntime();
  MACE_CHECK(runtime->GetRuntimeType() == RT_OPENCL);
  OpenclRuntime *opencl_runtime = static_cast<OpenclRuntime *>(runtime);
  auto *executor = opencl_runtime->GetOpenclExecutor();
  MACE_OUT_OF_RANGE_DEFINITION

  const uint32_t gws =
      static_cast<uint32_t>(RoundUpDiv4(output->size()));
  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_data_type");
    built_options.emplace("-Dtransform_data_type=" + kernel_name);
    built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
    MACE_RETURN_IF_ERROR(executor->BuildKernel("buffer_transform",
                                               kernel_name,
                                               built_options,
                                               kernel));
  }

  MACE_OUT_OF_RANGE_INITION(*kernel);
  uint32_t idx = 0;
  MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->size());
  kernel->setArg(idx++, gws);
  kernel->setArg(idx++, *(input->memory<cl::Buffer>()));
  MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
             "buffer offset not aligned");
  kernel->setArg(idx++,
                 static_cast<uint32_t>(input->buffer_offset() /
                     GetEnumTypeSize(input->dtype())));
  kernel->setArg(idx++, *(output->memory<cl::Buffer>()));

  const uint32_t lws =
      static_cast<uint32_t>(RoundUpDiv4(executor->GetDeviceMaxWorkGroupSize()));
  cl::Event event;
  cl_int error;
  if (executor->IsNonUniformWorkgroupsSupported()) {
    error = executor->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange, cl::NDRange(gws),
        cl::NDRange(lws), nullptr, &event);
  } else {
    uint32_t roundup_gws = RoundUp(gws, lws);
    error = executor->command_queue().enqueueNDRangeKernel(
        *kernel, cl::NullRange, cl::NDRange(roundup_gws),
        cl::NDRange(lws), nullptr, &event);
  }
  MACE_CL_RET_STATUS(error);
  MACE_OUT_OF_RANGE_VALIDATE;
  if (context->future() != nullptr) {
    context->future()->wait_fn = [executor, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        executor->GetCallStats(event, stats);
      }
    };
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace opencl
}  // namespace runtimes
}  // namespace mace

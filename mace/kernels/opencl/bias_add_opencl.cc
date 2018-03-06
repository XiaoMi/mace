//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/bias_add.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void BiasAddFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = OpenCLRuntime::Global();
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("bias_add");
    built_options.emplace("-Dbias_add=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    kernel_ = runtime->BuildKernel("bias_add", kernel_name, built_options);

    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(bias->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8};

  cl::Event event;
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      kernel_, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      nullptr, &event);
  MACE_CHECK(error == CL_SUCCESS);
  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
}

template
struct BiasAddFunctor<DeviceType::OPENCL, float>;
template
struct BiasAddFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

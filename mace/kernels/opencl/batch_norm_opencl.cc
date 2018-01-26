//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void BatchNormFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                         const Tensor *scale,
                                                         const Tensor *offset,
                                                         const Tensor *mean,
                                                         const Tensor *var,
                                                         const float epsilon,
                                                         Tensor *output,
                                                         StatsFuture *future) {
  MACE_CHECK(folded_constant_ || (mean != nullptr && var != nullptr));

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("batch_norm");
  built_options.emplace("-Dbatch_norm=" + kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  if (folded_constant_) {
    built_options.emplace("-DFOLDED_CONSTANT");
  }
  switch (activation_) {
    case NOOP:
      break;
    case RELU:
      built_options.emplace("-DUSE_RELU");
      break;
    case RELUX:
      built_options.emplace("-DUSE_RELUX");
      break;
    case PRELU:
      built_options.emplace("-DUSE_PRELU");
      break;
    case TANH:
      built_options.emplace("-DUSE_TANH");
      break;
    case SIGMOID:
      built_options.emplace("-DUSE_SIGMOID");
      break;
    defeult:
      LOG(FATAL) << "Unknown activation type: " << activation_;
  }

  auto bm_kernel =
      runtime->BuildKernel("batch_norm", kernel_name, built_options);

  uint32_t idx = 0;
  bm_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(scale->buffer())));
  bm_kernel.setArg(idx++,
                   *(static_cast<const cl::Image2D *>(offset->buffer())));
  if (!folded_constant_) {
    bm_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(mean->buffer())));
    bm_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(var->buffer())));
    bm_kernel.setArg(idx++, epsilon);
  }
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  bm_kernel.setArg(idx++, relux_max_limit_);
  bm_kernel.setArg(idx++, prelu_alpha_);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::string tuning_key =
      Concat("batch_norm_opencl_kernel_", activation_, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3), folded_constant_);
  TuningOrRun3DKernel(bm_kernel, tuning_key, gws, lws, future);
}

template struct BatchNormFunctor<DeviceType::OPENCL, float>;
template struct BatchNormFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

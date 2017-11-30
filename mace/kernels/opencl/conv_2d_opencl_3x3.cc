//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static void Conv2d3x3S12(const Tensor *input, const Tensor *filter,
                         const Tensor *bias, const uint32_t stride,
                         const int *padding, Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv<index_t, 5>(width);

  std::set<std::string> built_options;
  built_options.emplace(input->dtype() == DT_FLOAT ? "-DTYPE_FLOAT" : "");
  built_options.emplace("-DCMD_DATA_TYPE=" + DataTypeToOPENCLCMDDataType(input->dtype()));
  built_options.emplace(bias != nullptr ? "-DBIAS" : "");
  built_options.emplace("-DSTRIDE=" + ToString(stride));

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  auto conv_2d_kernel = runtime->BuildKernel("conv_2d_3x3", "conv_2d_3x3", built_options);
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(1)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(2)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));
  conv_2d_kernel.setArg(idx++, padding[0] / 2);
  conv_2d_kernel.setArg(idx++, padding[1] / 2);

  auto command_queue = runtime->command_queue();
  cl_int error;
  error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<uint32_t>(channel_blocks), static_cast<uint32_t>(width_blocks),
                  static_cast<uint32_t>(height * batch)),
      cl::NDRange(16, 16, 4),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);

}
void Conv2dOpenclK3x3S1(const Tensor *input, const Tensor *filter,
                        const Tensor *bias, const int *padding, Tensor *output) {
  Conv2d3x3S12(input, filter, bias, 1, padding, output);
};

void Conv2dOpenclK3x3S2(const Tensor *input, const Tensor *filter,
                        const Tensor *bias, const int *padding, Tensor *output) {
  Conv2d3x3S12(input, filter, bias, 2, padding, output);
};

}  // namespace kernels
}  // namespace mace

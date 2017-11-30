//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

void Conv1x1(const Tensor *input,
             const Tensor *filter,
             const Tensor *bias,
             const int stride,
             Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_blocks = RoundUpDiv4(width);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);

  MACE_CHECK(input_batch == batch);

  std::set<std::string> built_options;
  built_options.emplace(input->dtype() == DT_FLOAT ? "-DTYPE_FLOAT" : "");
  built_options.emplace("-DCMD_DATA_TYPE=" + DataTypeToOPENCLCMDDataType(input->dtype()));
  built_options.emplace("-DSTRIDE=" + ToString(stride));
  if (bias != nullptr) {
    built_options.emplace("-DBIAS");
  }

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  auto conv_2d_kernel = runtime->BuildKernel("conv_2d_1x1", "conv_2d_1x1", built_options);
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_width));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));

  auto command_queue = runtime->command_queue();
  cl_int error;
  error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<uint32_t>(channel_blocks),
                  static_cast<uint32_t>(width_blocks),
                  static_cast<uint32_t>(height * batch)),
      cl::NDRange(4, 15, 8), // TODO auto tuning
      nullptr, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);
}

extern void Conv2dOpenclK1x1S1(const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               const int *padding,
                               Tensor *output) {
  Conv1x1(input, filter, bias, 1, output);
};

extern void Conv2dOpenclK1x1S2(const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               const int *padding,
                               Tensor *output) {
  Conv1x1(input, filter, bias, 2, output);
};

}  // namespace kernels
}  // namespace mace

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

static void Pooling3(const Tensor *input,
                     const int *stride,
                     const PoolingType type,
                     Tensor *output) {
  if (type != MAX) {
    MACE_NOT_IMPLEMENTED;
  }
  index_t batch = output->dim(0);
  index_t channels = output->dim(1);
  index_t out_height = output->dim(2);
  index_t out_width = output->dim(3);

  index_t channel_blk = (channels + 3) / 4;
  const index_t pixel_width = (out_width + 3) / 4 ;
  const uint32_t gws[3] = {
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(channel_blk),
      static_cast<uint32_t>(pixel_width * out_height),
  };

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  built_options.emplace(stride[0] == 1 ? "-DSTRIDE_1" : "");
  auto pooling_kernel  = runtime->BuildKernel("pooling", "pooling3", built_options);


  const uint32_t lws[3] = {1, 8, 128};

  uint32_t idx = 0;
  pooling_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(2)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(3)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(channels));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(out_height));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(out_width));
  pooling_kernel.setArg(idx++, stride[0]);
  pooling_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      pooling_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS);
}

static void PoolingN(const Tensor *input,
                     const int *stride,
                     const int *paddings,
                     const int pooling_size,
                     const PoolingType type,
                     Tensor *output) {
  if (type != AVG) {
    MACE_NOT_IMPLEMENTED;
  }
  index_t batch = output->dim(0);
  index_t channels = output->dim(1);
  index_t out_height = output->dim(2);
  index_t out_width = output->dim(3);

  index_t channel_blk = (channels + 3) / 4;
  const uint32_t gws[3] = {
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(channel_blk),
      static_cast<uint32_t>(out_height * out_width),
  };

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  auto pooling_kernel  = runtime->BuildKernel("pooling", "poolingn", built_options);

  const uint32_t lws[3] = {1, 8, 128};

  uint32_t idx = 0;
  pooling_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(2)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(3)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(channels));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(out_height));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(out_width));
  pooling_kernel.setArg(idx++, stride[0]);
  pooling_kernel.setArg(idx++, paddings[0]);
  pooling_kernel.setArg(idx++, paddings[1]);
  pooling_kernel.setArg(idx++, pooling_size);
  pooling_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      pooling_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS);
}

template <>
void PoolingFunctor<DeviceType::OPENCL, float>::operator()(const Tensor *input,
                                                          Tensor *output) {
  int paddings[2];
  std::vector<index_t> filter_shape = {input->dim(1), input->dim(0),
                                       kernels_[0], kernels_[1]};
  kernels::CalPaddingSize(input->shape().data(), filter_shape.data(), this->dilations_,
                          strides_, this->padding_, paddings);
#define POOLING_HELPER                                               \
  switch(kernels_[0]) {                                              \
    case 3:                                                          \
      Pooling3(input, strides_, pooling_type_, output);              \
      break;                                                         \
    default:                                                         \
      PoolingN(input, strides_, paddings, kernels_[0],               \
               pooling_type_, output);                               \
      break;                                                         \
  }

  if (paddings[0] > 0 || paddings[1] > 0) {
    Tensor padded_input(GetDeviceAllocator(DeviceType::OPENCL), DataTypeToEnum<float>::v());
    ConstructInputWithPadding(input, paddings, &padded_input, pooling_type_ == MAX);
    input = &padded_input;
    POOLING_HELPER
  } else {
    POOLING_HELPER
  }
#undef POOLING_HELPER
}

}  // namespace kernels
}  // namespace mace

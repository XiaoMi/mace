//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

static void Pooling(const Tensor *input,
                    const int *stride,
                    const int *paddings,
                    const int pooling_size,
                    const PoolingType type,
                    const DataType dt,
                    Tensor *output) {
  index_t batch = output->dim(0);
  index_t out_height = output->dim(1);
  index_t out_width = output->dim(2);
  index_t channels = output->dim(3);

  index_t channel_blocks = (channels + 3) / 4;
  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blocks),
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(batch * out_height),
  };

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  if (type == MAX && input->dtype() == output->dtype()) {
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(dt == DT_HALF ? "-DFP16" : "");
  } else {
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  }
  if (type == AVG) {
    built_options.emplace("-DPOOL_AVG");
  }
  auto pooling_kernel = runtime->BuildKernel("pooling", "pooling", built_options);

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(pooling_kernel);

  uint32_t lws[3];
  lws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
  lws[1] = std::min<uint32_t>(out_width, kwg_size / lws[0]);
  lws[2] = std::min<uint32_t>(out_height * batch, kwg_size / (lws[0] * lws[1]));

  uint32_t idx = 0;
  pooling_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(1)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(input->dim(2)));
  pooling_kernel.setArg(idx++, static_cast<int32_t>(out_height));
  pooling_kernel.setArg(idx++, paddings[0] / 2);
  pooling_kernel.setArg(idx++, paddings[1] / 2);
  pooling_kernel.setArg(idx++, stride[0]);
  pooling_kernel.setArg(idx++, pooling_size);
  pooling_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      pooling_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS) << error;
}

template<typename T>
void PoolingFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                       Tensor *output) {
  MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1) << "Pooling opencl kernel not support dilation yet";
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  std::vector<index_t> filter_shape = {
      kernels_[0], kernels_[1],
      input->dim(3), input->dim(3)
  };

  kernels::CalcNHWCPaddingAndOutputSize(
      input->shape().data(), filter_shape.data(),
      dilations_, strides_, this->padding_,
      output_shape.data(), paddings.data());

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  Pooling(input, strides_, paddings.data(), kernels_[0], pooling_type_,
          DataTypeToEnum<T>::value, output);

}

template
struct PoolingFunctor<DeviceType::OPENCL, float>;
template
struct PoolingFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace

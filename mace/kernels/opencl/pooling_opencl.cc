//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

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

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blocks),
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(batch * out_height),
  };
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(pooling_kernel);
  std::vector<uint32_t> lws(3, 0);
  lws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
  lws[1] = std::min<uint32_t>(out_width, kwg_size / lws[0]);
  lws[2] = std::min<uint32_t>(out_height * batch, kwg_size / (lws[0] * lws[1]));
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(out_width, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(out_height * batch, kwg_size / (local_ws[0] * local_ws[1]));
    return {{4, 15, 8}, //SNPE size
            {local_ws[0], local_ws[1], local_ws[2]},
            {kwg_size / 16, 4, 4},
            {kwg_size / 32, 4, 8},
            {kwg_size / 32, 8, 4},
            {kwg_size / 64, 8, 8},
            {kwg_size / 64, 16, 4},
            {kwg_size / 128, 8, 16},
            {kwg_size / 128, 16, 8},
            {kwg_size / 128, 32, 4},
            {1, kwg_size / 32, 32},
            {1, kwg_size / 64, 64},
            {1, kwg_size / 128, 128},
            {3, 15, 9},
            {7, 15, 9},
            {9, 7, 15},
            {15, 7, 9},
            {1, kwg_size, 1}};
  };
  auto func = [&](const std::vector<uint32_t> &params) -> cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        pooling_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(params[0], params[1], params[2]),
        NULL, OpenCLRuntime::Get()->GetDefaultEvent());

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "pooling_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(ss.str(),
                                                     lws,
                                                     params_generator,
                                                     func);
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

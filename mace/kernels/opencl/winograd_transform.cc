//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/winograd_transform.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template<typename T>
void WinogradTransformFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input_tensor,
                                                                 Tensor *output_tensor,
                                                                 StatsFuture *future) {
  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {3, 3, input_tensor->dim(3), 1};
  std::vector<int> paddings(2);
  kernels::CalcNHWCPaddingAndOutputSize(
      input_tensor->shape().data(), filter_shape.data(), dilations_.data(),
      strides_.data(), paddings_, output_shape.data(), paddings.data());

  const index_t round_h = (output_shape[1] + 1) / 2;
  const index_t round_w = (output_shape[2] + 1) / 2;
  const index_t out_width = input_tensor->dim(0) * round_h * round_w;
  output_shape = {16, input_tensor->dim(3), out_width, 1};
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_HEIGHT, image_shape);
  output_tensor->ResizeImage(output_shape, image_shape);

  string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL("winograd_transform_2x2");
  std::set<std::string> built_options;
  built_options.emplace("-Dwinograd_transform_2x2=" + obfuscated_kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(DataTypeToEnum<T>::value));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
  auto runtime = OpenCLRuntime::Global();
  auto wino_kernel = runtime->BuildKernel("winograd_transform",
                                         obfuscated_kernel_name,
                                         built_options);

  uint32_t idx = 0;
  wino_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input_tensor->buffer())));
  wino_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output_tensor->buffer())));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(1)));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(2)));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(3)));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(round_h * round_w));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(round_w));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(paddings[0] / 2));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(paddings[1] / 2));

  const size_t gws[2] = {static_cast<size_t>(out_width),
                         static_cast<size_t>(RoundUpDiv4(input_tensor->dim(3)))};
  const std::vector<uint32_t> lws = {128, 8};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(wino_kernel);
  auto params_generator = [&]()->std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(2, 0);
    local_ws[0] = std::min<uint32_t>(gws[0], kwg_size);
    local_ws[1] = std::min<uint32_t>(gws[1], kwg_size / local_ws[0]);
    return {{local_ws[0], local_ws[1]},
            {local_ws[1], local_ws[0]},
            {kwg_size / 4, 4},
            {kwg_size / 8, 8},
            {kwg_size / 16, 16},
            {kwg_size / 32, 32},
            {kwg_size / 64, 64},
            {kwg_size / 128, 128},
            {kwg_size / 256, 256},
            {kwg_size / 512, 512},
            {kwg_size, 1},
            {1, kwg_size}
    };
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t>& params)->cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        wino_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1]),
        cl::NDRange(params[0], params[1]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "winograd_transform_kernel_"
     << input_tensor->dim(0) << "_"
     << input_tensor->dim(1) << "_"
     << input_tensor->dim(2) << "_"
     << input_tensor->dim(3);
  OpenCLProfilingTimer timer(&event);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(ss.str(),
                                                     lws,
                                                     params_generator,
                                                     func,
                                                     &timer);

  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
}

template<typename T>
void WinogradInverseTransformFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input_tensor,
                                                                        Tensor *output_tensor,
                                                                        StatsFuture *future) {
  std::vector<index_t> output_shape = {batch_, height_, width_, input_tensor->dim(1)};
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, image_shape);
  output_tensor->ResizeImage(output_shape, image_shape);

  string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL("winograd_inverse_transform_2x2");
  std::set<std::string> built_options;
  built_options.emplace("-Dwinograd_inverse_transform_2x2=" + obfuscated_kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(DataTypeToEnum<T>::value));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
  if ((input_tensor->dim(1) % 4 == 0 || input_tensor->dim(0) == 1) &&
      input_tensor->dim(2) % 4 == 0) {
    built_options.emplace("-DDIVISIBLE_FOUR");
  }
  auto runtime = OpenCLRuntime::Global();
  auto wino_kernel = runtime->BuildKernel("winograd_transform",
                                         obfuscated_kernel_name,
                                         built_options);

  const uint32_t round_h = (height_ + 1) / 2;
  const uint32_t round_w = (width_ + 1) / 2;
  uint32_t idx = 0;
  wino_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input_tensor->buffer())));
  wino_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output_tensor->buffer())));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(output_shape[1]));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(output_shape[2]));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(round_h * round_w));
  wino_kernel.setArg(idx++, static_cast<uint32_t>(round_w));

  const size_t gws[2] = {static_cast<size_t>(input_tensor->dim(2)),
                         static_cast<size_t>(RoundUpDiv4(input_tensor->dim(1)))};
  const std::vector<uint32_t> lws = {128, 8};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(wino_kernel);
  auto params_generator = [&]()->std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(2, 0);
    local_ws[0] = std::min<uint32_t>(gws[0], kwg_size);
    local_ws[1] = std::min<uint32_t>(gws[1], kwg_size / local_ws[0]);
    return {{local_ws[0], local_ws[1]},
            {local_ws[1], local_ws[0]},
            {kwg_size / 4, 4},
            {kwg_size / 8, 8},
            {kwg_size / 16, 16},
            {kwg_size / 32, 32},
            {kwg_size / 64, 64},
            {kwg_size / 128, 128},
            {kwg_size / 256, 256},
            {kwg_size / 512, 512},
            {kwg_size, 1},
            {1, kwg_size}
    };
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t>& params)->cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        wino_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1]),
        cl::NDRange(params[0], params[1]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "winograd_inverse_transform_kernel_"
     << input_tensor->dim(0) << "_"
     << input_tensor->dim(1) << "_"
     << input_tensor->dim(2) << "_"
     << input_tensor->dim(3);
  OpenCLProfilingTimer timer(&event);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(ss.str(),
                                                     lws,
                                                     params_generator,
                                                     func,
                                                     &timer);

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
struct WinogradTransformFunctor<DeviceType::OPENCL, float>;
template
struct WinogradTransformFunctor<DeviceType::OPENCL, half>;

template
struct WinogradInverseTransformFunctor<DeviceType::OPENCL, float>;
template
struct WinogradInverseTransformFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

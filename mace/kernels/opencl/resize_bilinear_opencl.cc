//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/kernels/resize_bilinear.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void ResizeBilinearFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input, Tensor *output, StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  index_t out_height = out_height_;
  index_t out_width = out_width_;
  MACE_CHECK(out_height > 0 && out_width > 0);
  std::vector<index_t> output_shape {batch, out_height, out_width, channels};
  if (input->is_image()) {
    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
    output->ResizeImage(output_shape, output_image_shape);
  } else {
    output->Resize(output_shape);
  }

  float height_scale =
      CalculateResizeScale(in_height, out_height, align_corners_);
  float width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("resize_bilinear_nocache");
  built_options.emplace("-Dresize_bilinear_nocache=" + kernel_name);
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  auto rb_kernel  = runtime->BuildKernel("resize_bilinear", kernel_name, built_options);

  uint32_t idx = 0;
  rb_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  rb_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  rb_kernel.setArg(idx++, height_scale);
  rb_kernel.setArg(idx++, width_scale);
  rb_kernel.setArg(idx++, static_cast<int32_t>(in_height));
  rb_kernel.setArg(idx++, static_cast<int32_t>(in_width));
  rb_kernel.setArg(idx++, static_cast<int32_t>(out_height));

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(out_width),
                           static_cast<uint32_t>(out_height * batch)};
  std::vector<uint32_t> lws = {8, 16, 8, 1};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(rb_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(out_width, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(out_height * batch, kwg_size / (local_ws[0] * local_ws[1]));
    return {
        {local_ws[0], local_ws[1], local_ws[2], 1},
        {kwg_size / 16, 4, 4, 1},
        {kwg_size / 32, 4, 8, 1},
        {kwg_size / 32, 8, 4, 1},
        {kwg_size / 64, 8, 8, 1},
        {kwg_size / 64, 16, 4, 1},
        {kwg_size / 128, 8, 16, 1},
        {kwg_size / 128, 16, 8, 1},
        {kwg_size / 128, 32, 4, 1},
        {1, kwg_size / 32, 32, 1},
        {1, kwg_size / 64, 64, 1},
        {1, kwg_size / 128, 128, 1},
        {3, 15, 9, 1},
        {7, 15, 9, 1},
        {9, 7, 15, 1},
        {15, 7, 9, 1},
        {1, kwg_size, 1, 1},
        {4, 15, 8, 1},  // SNPE size
    };
  };
  cl::Event event;
  auto func = [&](std::vector<uint32_t> &params, Timer *timer) -> cl_int {
    cl_int error = CL_SUCCESS;
    if (timer == nullptr) {
      uint32_t num_blocks = params.back();
      const uint32_t block_size = gws[2] / num_blocks;
      if (gws[2] % num_blocks > 0) num_blocks++;
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws2 = (i == num_blocks - 1) ? (gws[2] - (i * block_size)) : block_size;
        error = runtime->command_queue().enqueueNDRangeKernel(
            rb_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      }
    } else {
      timer->StartTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          rb_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
          cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
      MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      timer->StopTiming();
      double elapse_time = timer->ElapsedMicros();
      timer->ClearTiming();
      uint32_t num_blocks = std::min(static_cast<uint32_t>(elapse_time / kMaxKernelExeTime) + 1, gws[2]);
      params.back() = num_blocks;
      const uint32_t block_size = gws[2] / num_blocks;
      if (gws[2] % num_blocks > 0) num_blocks++;
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws2 = (i == num_blocks - 1) ? (gws[2] - (i * block_size)) : block_size;
        error = runtime->command_queue().enqueueNDRangeKernel(
            rb_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
        timer->AccumulateTiming();
      }
    }
    return error;
  };
  std::stringstream ss;
  ss << "resize_bilinear_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
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

template struct ResizeBilinearFunctor<DeviceType::OPENCL, float>;
template struct ResizeBilinearFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

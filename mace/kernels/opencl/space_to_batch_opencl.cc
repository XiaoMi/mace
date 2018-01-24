//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_
#define MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/space_to_batch.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void SpaceToBatchFunctor<DeviceType::OPENCL, T>::operator()(Tensor *space_tensor,
                                                            const std::vector<index_t> &output_shape,
                                                            Tensor *batch_tensor,
                                                            StatsFuture *future) {
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
  const char *kernel_name = nullptr;
  if (b2s_) {
    space_tensor->ResizeImage(output_shape, output_image_shape);
    kernel_name = "batch_to_space";
  } else {
    batch_tensor->ResizeImage(output_shape, output_image_shape);
    kernel_name = "space_to_batch";
  }
  std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  std::stringstream kernel_name_ss;
  kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
  built_options.emplace(kernel_name_ss.str());
  built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DataTypeToEnum<T>::value));
  auto s2b_kernel = runtime->BuildKernel("space_to_batch", kernel_name, built_options);

  uint32_t idx = 0;
  if (b2s_) {
    s2b_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(batch_tensor->buffer())));
    s2b_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(space_tensor->buffer())));
  } else {
    s2b_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(space_tensor->buffer())));
    s2b_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(batch_tensor->buffer())));
  }
  s2b_kernel.setArg(idx++, block_shape_[0]);
  s2b_kernel.setArg(idx++, block_shape_[1]);
  s2b_kernel.setArg(idx++, paddings_[0]);
  s2b_kernel.setArg(idx++, paddings_[2]);
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(1)));
  s2b_kernel.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));

  const uint32_t chan_blk = RoundUpDiv4<uint32_t>(batch_tensor->dim(3));
  const uint32_t gws[3] = {chan_blk,
                           static_cast<uint32_t>(batch_tensor->dim(2)),
                           static_cast<uint32_t>(batch_tensor->dim(0) * batch_tensor->dim(1))};
  std::vector<uint32_t> lws = {8, 16, 8, 1};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(s2b_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(chan_blk, kwg_size);
    local_ws[1] = std::min<uint32_t>(32, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(32, kwg_size / (local_ws[0] * local_ws[1]));
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
            s2b_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      }
    } else {
      timer->StartTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          s2b_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
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
            s2b_kernel,
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
  ss << kernel_name << "_"
     << batch_tensor->dim(0) << "_"
     << batch_tensor->dim(1) << "_"
     << batch_tensor->dim(2) << "_"
     << batch_tensor->dim(3);
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

template struct SpaceToBatchFunctor<DeviceType::OPENCL, float>;
template struct SpaceToBatchFunctor<DeviceType::OPENCL, half>;

} //  namespace kernels
} //  namespace mace
#endif //  MACE_KERNELS_OPENCL_SPACE_TO_BATCH_H_

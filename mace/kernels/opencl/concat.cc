//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/concat.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

static void Concat2(const Tensor *input0,
                    const Tensor *input1,
                    const DataType dt,
                    Tensor *output,
                    StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const int channel_blk = RoundUpDiv4(channel);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel");
  built_options.emplace("-Dconcat_channel=" + kernel_name);
  if (input0->dtype() == output->dtype()) {
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
  } else {
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  }
  if (input0->dim(3) % 4 == 0) {
    built_options.emplace("-DDIVISIBLE_FOUR");
  }
  auto concat_kernel = runtime->BuildKernel("concat", kernel_name, built_options);

  uint32_t idx = 0;
  concat_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input0->buffer())));
  concat_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input1->buffer())));
  concat_kernel.setArg(idx++, static_cast<int32_t>(input0->dim(3)));
  concat_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk),
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(batch * height),
  };
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(concat_kernel);
  std::vector<uint32_t> lws = {8, 16, 8, 1};
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blk, kwg_size);
    local_ws[1] = std::min<uint32_t>(width, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(height * batch, kwg_size / (local_ws[0] * local_ws[1]));
    return {{local_ws[0], local_ws[1], local_ws[2], 1},
            {local_ws[2], local_ws[1], local_ws[0], 1},
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
            {4, 15, 8, 1}, //SNPE size
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
            concat_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      }
    } else {
      timer->StartTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          concat_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
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
            concat_kernel,
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
  ss << "concat_opencl_kernel_"
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

template<typename T>
void ConcatFunctor<DeviceType::OPENCL, T>::operator()(const std::vector<const Tensor *> &input_list,
                                                      Tensor *output,
                                                      StatsFuture *future) {
  const int inputs_count = input_list.size();
  MACE_CHECK(inputs_count == 2 && axis_ == 3)
    << "Concat opencl kernel only support two elements with axis == 3";

  const Tensor *input0 = input_list[0];

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis_) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis_] += input->dim(axis_);
  }
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, image_shape);
  output->ResizeImage(output_shape, image_shape);

  switch (inputs_count) {
    case 2:
      Concat2(input_list[0], input_list[1], DataTypeToEnum<T>::value,
              output, future);
      break;
    default:MACE_NOT_IMPLEMENTED;
  }
};

template
struct ConcatFunctor<DeviceType::OPENCL, float>;
template
struct ConcatFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
} //  namespace mace

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
static void AddN(const std::vector<const Tensor *> &input_tensors,
                 Tensor *output, StatsFuture *future) {
  if (input_tensors.size() > 4) {
    MACE_NOT_IMPLEMENTED;
  }
  output->ResizeLike(input_tensors[0]);

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  std::string kernel_name = MACE_KERNRL_NAME("addn");
  built_options.emplace("-Daddn=" + kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  built_options.emplace("-DINPUT_NUM=" + ToString(input_tensors.size()));
  auto addn_kernel = runtime->BuildKernel("addn", kernel_name, built_options);

  uint32_t idx = 0;
  for (auto input : input_tensors) {
    addn_kernel.setArg(idx++,
                       *(static_cast<const cl::Image2D *>(input->buffer())));
  }
  addn_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_pixels),
      static_cast<uint32_t>(batch_height_pixels)
  };
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(addn_kernel);
  std::vector<uint32_t> lws = {64, 16};
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    uint32_t local_ws[2];
    local_ws[0] = std::min<uint32_t>(width_pixels, kwg_size);
    local_ws[1] = std::min<uint32_t>(batch_height_pixels, kwg_size / local_ws[0]);
    return {{local_ws[0], local_ws[1]},
            {local_ws[1], local_ws[0]},
            {kwg_size / 4, 4},
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
  auto func = [&](const std::vector<uint32_t> &params) -> cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        addn_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1]),
        cl::NDRange(params[0], params[1]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "addn_opencl_kernel_"
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

template <typename T>
void AddNFunctor<DeviceType::OPENCL, T>::operator()(
    const std::vector<const Tensor *> &input_tensors,
    Tensor *output_tensor,
    StatsFuture *future) {
  size_t size = input_tensors.size();
  MACE_CHECK(size >= 2 && input_tensors[0] != nullptr);

  const index_t batch = input_tensors[0]->dim(0);
  const index_t height = input_tensors[0]->dim(1);
  const index_t width = input_tensors[0]->dim(2);
  const index_t channels = input_tensors[0]->dim(3);

  for (int i = 1; i < size; ++i) {
    MACE_CHECK_NOTNULL(input_tensors[i]);
    MACE_CHECK(batch == input_tensors[i]->dim(0));
    MACE_CHECK(height == input_tensors[i]->dim(1));
    MACE_CHECK(width == input_tensors[i]->dim(2));
    MACE_CHECK(channels == input_tensors[i]->dim(3));
  }

  std::vector<index_t> output_shape = input_tensors[0]->shape();
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
  output_tensor->ResizeImage(output_shape, output_image_shape);

  AddN<T>(input_tensors, output_tensor, future);
};

template
struct AddNFunctor<DeviceType::OPENCL, float>;

template
struct AddNFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  //  namespace mace

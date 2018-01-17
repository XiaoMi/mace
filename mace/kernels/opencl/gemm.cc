//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/gemm.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void GEMMFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *A,
    const Tensor *B,
    Tensor *C,
    StatsFuture *future) {

  std::vector<index_t> c_shape = {A->dim(0), A->dim(1), 1, B->dim(3)};
  std::vector<size_t> c_image_shape;
  CalImage2DShape(c_shape, BufferType::IN_OUT, c_image_shape);
  C->ResizeImage(c_shape, c_image_shape);

  const index_t batch = C->dim(0);
  const index_t height = C->dim(1);
  const index_t width = C->dim(3);

  const index_t width_blocks = RoundUpDiv4(width);
  const index_t height_blocks = RoundUpDiv4(height);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("gemm");
  built_options.emplace("-Dgemm=" + kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  auto gemm_kernel = runtime->BuildKernel("gemm", kernel_name, built_options);

  uint32_t idx = 0;
  gemm_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(A->buffer())));
  gemm_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(B->buffer())));
  gemm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(C->buffer())));
  gemm_kernel.setArg(idx++, static_cast<int>(height));
  gemm_kernel.setArg(idx++, static_cast<int>(height_blocks));
  gemm_kernel.setArg(idx++, static_cast<int>(A->dim(3)));

  const uint32_t gws[3] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };
  const std::vector<uint32_t> lws = {16, 64};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(gemm_kernel);
  auto params_generator = [&]()->std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(2, 0);
    local_ws[0] = std::min<uint32_t>(width_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(height_blocks * batch, kwg_size / local_ws[0]);
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
  auto func = [&](const std::vector<uint32_t>& params)->cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        gemm_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1]),
        cl::NDRange(params[0], params[1]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "gemm_opencl_kernel_"
     << C->dim(0) << "_"
     << C->dim(1) << "_"
     << C->dim(2) << "_"
     << C->dim(3);
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

};

template
struct GEMMFunctor<DeviceType::OPENCL, float>;

template
struct GEMMFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  //  namespace mace

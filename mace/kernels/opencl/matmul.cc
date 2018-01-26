//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/matmul.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void MatMulFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *A,
    const Tensor *B,
    Tensor *C,
    StatsFuture *future) {

  std::vector<index_t> c_shape = {A->dim(0), A->dim(1), B->dim(2), 1};
  std::vector<size_t> c_image_shape;
  CalImage2DShape(c_shape, BufferType::IN_OUT_HEIGHT, c_image_shape);
  C->ResizeImage(c_shape, c_image_shape);

  const index_t batch = C->dim(0);
  const index_t height = C->dim(1);
  const index_t width = C->dim(2);

  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
  built_options.emplace("-Dmatmul=" + kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  auto matmul_kernel = runtime->BuildKernel("matmul", kernel_name, built_options);

  uint32_t idx = 0;
  matmul_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(A->buffer())));
  matmul_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(B->buffer())));
  matmul_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(C->buffer())));
  matmul_kernel.setArg(idx++, static_cast<int>(height));
  matmul_kernel.setArg(idx++, static_cast<int>(width));
  matmul_kernel.setArg(idx++, static_cast<int>(A->dim(2)));
  matmul_kernel.setArg(idx++, static_cast<int>(height_blocks));
  matmul_kernel.setArg(idx++, static_cast<int>(RoundUpDiv4(A->dim(2))));

  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };
  const std::vector<uint32_t> lws = {16, 64, 1};
  std::stringstream ss;
  ss << "matmul_opencl_kernel_"
     << C->dim(0) << "_"
     << C->dim(1) << "_"
     << C->dim(2) << "_"
     << C->dim(3);
  TuningOrRun2DKernel(matmul_kernel, ss.str(), gws, lws, future);

};

template
struct MatMulFunctor<DeviceType::OPENCL, float>;

template
struct MatMulFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  //  namespace mace

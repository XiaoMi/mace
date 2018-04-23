// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/kernels/matmul.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void MatMulFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *A,
                                                      const Tensor *B,
                                                      Tensor *C,
                                                      StatsFuture *future) {
  std::vector<index_t> c_shape = {A->dim(0), A->dim(1), B->dim(2), 1};
  std::vector<size_t> c_image_shape;
  CalImage2DShape(c_shape, BufferType::IN_OUT_HEIGHT, &c_image_shape);
  C->ResizeImage(c_shape, c_image_shape);

  const index_t batch = C->dim(0);
  const index_t height = C->dim(1);
  const index_t width = C->dim(2);

  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
    built_options.emplace("-Dmatmul=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    if (runtime->IsOutOfRangeCheckEnabled()) {
      built_options.emplace("-DOUT_OF_RANGE_CHECK");
      kernel_error_ = std::move(std::unique_ptr<Buffer>(
            new Buffer(GetDeviceAllocator(DeviceType::OPENCL), 1)));
      kernel_error_->Map(nullptr);
      *(kernel_error_->mutable_data<char>()) = 0;
      kernel_error_->UnMap();
    }
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
    kernel_ = runtime->BuildKernel("matmul", kernel_name, built_options);

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  uint32_t idx = 0;
  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_.setArg(idx++,
        *(static_cast<cl::Buffer *>(kernel_error_->buffer())));
  }
  if (!runtime->IsNonUniformWorkgroupsSupported()) {
    kernel_.setArg(idx++, gws[0]);
    kernel_.setArg(idx++, gws[1]);
  }
  kernel_.setArg(idx++, *(A->opencl_image()));
  kernel_.setArg(idx++, *(B->opencl_image()));
  kernel_.setArg(idx++, *(C->opencl_image()));
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<int>(width));
  kernel_.setArg(idx++, static_cast<int>(A->dim(2)));
  kernel_.setArg(idx++, static_cast<int>(height_blocks));
  kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(A->dim(2))));

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::stringstream ss;
  ss << "matmul_opencl_kernel_" << C->dim(0) << "_" << C->dim(1) << "_"
     << C->dim(2) << "_" << C->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);

  if (runtime->IsOutOfRangeCheckEnabled()) {
    kernel_error_->Map(nullptr);
    char *kerror_code = kernel_error_->mutable_data<char>();
    MACE_CHECK(*kerror_code == 0) << "Kernel error code: " << *kerror_code;
    kernel_error_->UnMap();
  }
}

template struct MatMulFunctor<DeviceType::OPENCL, float>;

template struct MatMulFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace

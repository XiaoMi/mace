// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#ifndef MACE_OPS_OPENCL_IMAGE_MATMUL_H_
#define MACE_OPS_OPENCL_IMAGE_MATMUL_H_

#include "mace/ops/opencl/matmul.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class MatMulKernel : public OpenCLMatMulKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *A,
      const Tensor *B,
      Tensor *C,
      bool transpose_a,
      bool transpose_b) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
};

template <typename T>
MaceStatus MatMulKernel<T>::Compute(
    OpContext *context,
    const Tensor *A,
    const Tensor *B,
    Tensor *C,
    bool transpose_a,
    bool transpose_b) {
  MACE_CHECK(!transpose_a && !transpose_b,
             "GPU does not support transpose matmul");

  index_t rank = A->dim_size();
  index_t height = A->dim(rank - 2);
  index_t K = A->dim(rank - 1);
  index_t width = B->dim(rank - 1);
  index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                  std::multiplies<index_t>());

  std::vector<index_t> c_shape = A->shape();
  c_shape[rank - 2] = height;
  c_shape[rank - 1] = width;
  std::vector<size_t> c_image_shape;
  std::vector<index_t> padded_c_shape = {batch, height, width, 1};
  OpenCLUtil::CalImage2DShape(padded_c_shape,
                              OpenCLBufferType::IN_OUT_HEIGHT,
                              &c_image_shape);
  MACE_RETURN_IF_ERROR(C->ResizeImage(c_shape, c_image_shape));

  const index_t height_blocks = RoundUpDiv4(height);
  const index_t width_blocks = RoundUpDiv4(width);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * batch),
  };

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("matmul");
    built_options.emplace("-Dmatmul=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("matmul", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
  MACE_SET_2D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *(A->opencl_image()));
  kernel_.setArg(idx++, *(B->opencl_image()));
  kernel_.setArg(idx++, *(C->opencl_image()));
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<int>(width));
  kernel_.setArg(idx++, static_cast<int>(K));
  kernel_.setArg(idx++, static_cast<int>(height_blocks));
  kernel_.setArg(idx++, static_cast<int>(RoundUpDiv4(K)));

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::string tuning_key = Concat("matmul_opencl_kernel", batch, height, width);
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_MATMUL_H_

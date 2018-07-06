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

#ifndef MACE_KERNELS_CROP_H_
#define MACE_KERNELS_CROP_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct CropFunctorBase {
  CropFunctorBase(const int axis,
                  const std::vector<int> &offset)
      : axis_(axis),
        offset_(offset) {}

  const int axis_;
  std::vector<int> offset_;
};

template <DeviceType D, typename T>
struct CropFunctor : CropFunctorBase {
  CropFunctor(const int axis, const std::vector<int> &offset)
      : CropFunctorBase(axis, offset) {}

  void crop_copy(const T* input_data, T* output_data,
                 const std::vector<index_t> &input_shape,
                 const std::vector<index_t> &output_shape,
                 const int32_t* offsets) {
    const index_t out_img_size =
        output_shape[1] * output_shape[2] * output_shape[3];
    const index_t out_hw = output_shape[2] * output_shape[3];
    const index_t in_img_size =
        input_shape[1] * input_shape[2] * input_shape[3];
    const index_t in_hw = input_shape[2] * input_shape[3];
#pragma omp parallel for collapse(3)
    for (int b = 0; b < output_shape[0]; ++b) {
      for (int c = 0; c < output_shape[1]; ++c) {
        for (int h = 0; h < output_shape[2]; ++h) {
          T* out_ptr =
              output_data + b * out_img_size + c * out_hw + h * output_shape[3];
          const T* in_ptr_bch =
              input_data + (b + offsets[0]) * in_img_size +
                  (c + offsets[1]) * in_hw +
                  (h + offsets[2]) * input_shape[3] + offsets[3];
          memcpy(out_ptr, in_ptr_bch,
                 output_shape[3] * sizeof(T));
        }
      }
    }
  }

  MaceStatus operator()(const std::vector<const Tensor *> &input_list,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);

    MACE_CHECK(input_list.size() == 2, "Crop op needs two inputs.");
    const Tensor *input0 = input_list[0];
    const Tensor *input1 = input_list[1];
    const uint32_t in0_dims = static_cast<uint32_t >(input0->dim_size());
    const uint32_t in1_dims = static_cast<uint32_t >(input0->dim_size());

    MACE_CHECK(in0_dims == 4 && in1_dims == 4,
               "crop op only supports 4-dims inputs now.");

    std::vector<int32_t> offsets(in0_dims, 0);

    std::vector<index_t> output_shape(input0->shape());
    for (index_t i = 0; i < in0_dims; ++i) {
      int32_t crop_offset = 0;
      index_t new_size = input0->dim(i);
      if (i >= axis_) {
        new_size = input1->dim(i);
        if (offset_.size() == 1) {
          crop_offset = offset_[0];
        } else if (offset_.size() > 1) {
          crop_offset = offset_[i - axis_];
        }
        MACE_CHECK(input0->dim(i) - crop_offset >= input1->dim(i))
        << "the crop for dimension" << i << "is out of bound with size"
        << input1->dim(i) << "and offset" << crop_offset;
      }
      output_shape[i] = new_size;
      offsets[i] = crop_offset;
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    T *output_data = output->mutable_data<T>();

    const T * input_data = input0->data<T>();

    crop_copy(input_data, output_data, input0->shape(),
              output_shape, offsets.data());

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct CropFunctor<DeviceType::GPU, T> : CropFunctorBase {
  CropFunctor(const int axis, const std::vector<int> &offset)
  : CropFunctorBase(axis, offset) {}

  MaceStatus operator()(const std::vector<const Tensor *> &input_list,
                  Tensor *output,
                  StatsFuture *future);
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CROP_H_

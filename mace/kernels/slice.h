//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_SLICE_H_
#define MACE_KERNELS_SLICE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct SliceFunctor {
  void operator()(const Tensor *input,
                  const std::vector<Tensor *> &output_list,
                  StatsFuture *future) {
    const index_t outer_size = input->dim(0) * input->dim(1) * input->dim(2);
    const index_t input_channels = input->dim(3);
    const size_t outputs_count = output_list.size();
    const index_t output_channels = input_channels / outputs_count;
    std::vector<T *> output_ptrs(output_list.size(), nullptr);

    std::vector<index_t> output_shape({input->dim(0), input->dim(1),
                                       input->dim(2), output_channels});

    for (size_t i= 0; i < outputs_count; ++i) {
      output_list[i]->Resize(output_shape);
      output_ptrs[i] = output_list[i]->mutable_data<T>();
    }
    const T *input_ptr = input->data<T>();

#pragma omp parallel for
    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      int input_idx = outer_idx * input_channels;
      int output_idx = outer_idx * output_channels;
      for (size_t i = 0; i < outputs_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptrs[i]+output_idx, input_ptr+input_idx,
                 output_channels * sizeof(T));
        } else {
          for (index_t k = 0; k < output_channels; ++k) {
            *(output_ptrs[i] + output_idx + k) = *(input_ptr + input_idx + k);
          }
        }
        input_idx += output_channels;
      }
    }
  }
};

template<typename T>
struct SliceFunctor<DeviceType::OPENCL, T> {
  void operator()(const Tensor *input,
                  const std::vector<Tensor *> &output_list,
                  StatsFuture *future);
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SLICE_H_

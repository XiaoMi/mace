//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONCAT_H_
#define MACE_KERNELS_CONCAT_H_

#include "mace/core/common.h"
#include "mace/core/types.h"
#include "mace/core/mace.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

struct ConcatFunctorBase {
  ConcatFunctorBase(const int32_t axis): axis_(axis){}

  int32_t axis_;
};

template<DeviceType D, typename T>
struct ConcatFunctor : ConcatFunctorBase {
  ConcatFunctor(const int32_t axis): ConcatFunctorBase(axis){}

  void operator()(const std::vector<const Tensor *> &input_list,
                  Tensor *output) {
    const Tensor *input0 = input_list.front();
    const int inputs_count = input_list.size();

    std::vector<index_t> output_shape(input0->shape());
    index_t inner_size = 1;
    for (int i = 0; i < axis_; ++i) {
      inner_size *= output_shape[i];
    }
    std::vector<index_t> outer_sizes(inputs_count, 0);
    outer_sizes[0] = input0->size() / inner_size;
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
      outer_sizes[i] = input->size() / inner_size;
      output_shape[axis_] += input->dim(axis_);
    }
    output->Resize(output_shape);

    T *output_ptr = output->mutable_data<T>();

    std::vector<const T *> input_ptrs(input_list.size(), nullptr);
    for (size_t i = 0; i < inputs_count; ++i) {
      input_ptrs[i] = input_list[i]->data<T>();
    }
    for (int inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      for (size_t i = 0; i < inputs_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptr, input_ptrs[i], outer_sizes[i] * sizeof(T));
          output_ptr += outer_sizes[i];
          input_ptrs[i] += outer_sizes[i];
        } else {
          for (index_t k = 0; k < outer_sizes[i]; ++k) {
            *output_ptr++ = *input_ptrs[i]++;
          }
        }
      }
    }
  }
};

template<typename T>
struct ConcatFunctor<DeviceType::OPENCL, T> : ConcatFunctorBase{
  ConcatFunctor(const int32_t axis): ConcatFunctorBase(axis){}

  void operator()(const std::vector<const Tensor *> &input_list,
                  Tensor *output);

};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_CONCAT_H_

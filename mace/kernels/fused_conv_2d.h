//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_FUSED_CONV_2D_H_
#define MACE_KERNELS_FUSED_CONV_2D_H_

#include "mace/core/tensor.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/conv_2d.h"

namespace mace {
namespace kernels {

struct FusedConv2dFunctorBase {
  FusedConv2dFunctorBase(const int *strides,
                         const Padding &paddings,
                         const int *dilations)
      : strides_(strides), dilations_(dilations), paddings_(paddings) {}

  const int *strides_;         // [stride_h, stride_w]
  const int *dilations_;       // [dilation_h, dilation_w]
  Padding paddings_;
};

template<DeviceType D, typename T>
struct FusedConv2dFunctor : FusedConv2dFunctorBase {
  FusedConv2dFunctor(const int *strides,
                     const Padding &paddings,
                     const int *dilations)
      : FusedConv2dFunctorBase(strides, paddings, dilations) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    Conv2dFunctor<D, T>(strides_, paddings_, dilations_)(input, filter, bias,
                                                         output, future);
    T *output_data = output->mutable_data<T>();

    T zero_value;
    if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
      zero_value = half_float::half_cast<half>(0.0f);
    } else {
      zero_value = 0;
    }
    auto output_size = output->size();
    for (int n = 0; n < output_size; ++n) {
      *output_data = *output_data < 0 ? zero_value : *output_data;
      output_data++;
    }
  }

};

template<typename T>
struct FusedConv2dFunctor<DeviceType::OPENCL, T> : FusedConv2dFunctorBase {
  FusedConv2dFunctor(const int *strides,
                     const Padding &paddings,
                     const int *dilations)
      : FusedConv2dFunctorBase(strides, paddings, dilations) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_FUSED_CONV_2D_H_

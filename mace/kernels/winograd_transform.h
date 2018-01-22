//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_WINOGRAD_TRANSFORM_H_
#define MACE_KERNELS_WINOGRAD_TRANSFORM_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

struct WinogradTransformFunctorBase {
  WinogradTransformFunctorBase(const Padding &paddings)
      : strides_({1, 1}), dilations_({1, 1}), paddings_(paddings) {}

  const std::vector<int> strides_;         // [stride_h, stride_w]
  const std::vector<int> dilations_;       // [dilation_h, dilation_w]
  Padding paddings_;
};

template<DeviceType D, typename T>
struct WinogradTransformFunctor : WinogradTransformFunctorBase {
  WinogradTransformFunctor(const Padding &paddings)
      : WinogradTransformFunctorBase(paddings) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }

};

template<typename T>
struct WinogradTransformFunctor<DeviceType::OPENCL, T> : WinogradTransformFunctorBase {
  WinogradTransformFunctor(const Padding &paddings)
      : WinogradTransformFunctorBase(paddings) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);
};

struct WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctorBase(const int batch,
                                      const int height,
                                      const int width)
      : batch_(batch), height_(height), width_(width) {}

  const int batch_;
  const int height_;
  const int width_;
};

template<DeviceType D, typename T>
struct WinogradInverseTransformFunctor : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const int batch,
                                  const int height,
                                  const int width)
      : WinogradInverseTransformFunctorBase(batch, height, width) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }

};

template<typename T>
struct WinogradInverseTransformFunctor<DeviceType::OPENCL, T> : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const int batch,
                                  const int height,
                                  const int width)
      : WinogradInverseTransformFunctorBase(batch, height, width) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_WINOGRAD_TRANSFORM_H_

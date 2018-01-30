//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_WINOGRAD_TRANSFORM_H_
#define MACE_KERNELS_WINOGRAD_TRANSFORM_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/activation.h"
#include "mace/core/runtime/opencl/cl2_header.h"

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

  cl::Kernel kernel_;
};

struct WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctorBase(const int batch,
                                      const int height,
                                      const int width,
                                      const ActivationType activation,
                                      const float relux_max_limit,
                                      const float prelu_alpha)
      : batch_(batch),
        height_(height),
        width_(width),
        activation_(activation),
        relux_max_limit_(relux_max_limit),
        prelu_alpha_(prelu_alpha) {}

  const int batch_;
  const int height_;
  const int width_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float prelu_alpha_;
};

template<DeviceType D, typename T>
struct WinogradInverseTransformFunctor : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const int batch,
                                  const int height,
                                  const int width,
                                  const ActivationType activation,
                                  const float relux_max_limit,
                                  const float prelu_alpha)
      : WinogradInverseTransformFunctorBase(batch, height, width, activation, relux_max_limit, prelu_alpha) {}

  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }

};

template<typename T>
struct WinogradInverseTransformFunctor<DeviceType::OPENCL, T> : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const int batch,
                                  const int height,
                                  const int width,
                                  const ActivationType activation,
                                  const float relux_max_limit,
                                  const float prelu_alpha)
      : WinogradInverseTransformFunctorBase(batch, height, width, activation, relux_max_limit, prelu_alpha) {}

  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_WINOGRAD_TRANSFORM_H_

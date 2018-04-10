//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BUFFER_TO_IMAGE_H_
#define MACE_KERNELS_BUFFER_TO_IMAGE_H_

#include <memory>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

struct BufferToImageFunctorBase {
  explicit BufferToImageFunctorBase(bool i2b)
    : i2b_(i2b), kernel_error_(nullptr) {}
  bool i2b_;
  std::unique_ptr<BufferBase> kernel_error_;
};

template <DeviceType D, typename T>
struct BufferToImageFunctor : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(bool i2b = false)
      : BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }
};

template <typename T>
struct BufferToImageFunctor<DeviceType::OPENCL, T> : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(bool i2b = false)
      : BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BUFFER_TO_IMAGE_H_

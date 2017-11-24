//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#include "mace/core/tensor.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

struct BufferToImageFunctorBase {
  BufferToImageFunctorBase(bool i2b) : i2b_(i2b) {}
  bool i2b_;
};

template<DeviceType D, typename T>
struct BufferToImageFunctor : BufferToImageFunctorBase{
  BufferToImageFunctor(bool i2b = false) :
      BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output) {
    MACE_NOT_IMPLEMENTED;
  }
  bool i2b_;
};

template<typename T>
struct BufferToImageFunctor<DeviceType::OPENCL, T> : BufferToImageFunctorBase{
  BufferToImageFunctor(bool i2b = false) :
      BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output);
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BATCH_NORM_H_

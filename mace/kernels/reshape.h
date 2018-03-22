//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_RESHAPE_H_
#define MACE_KERNELS_RESHAPE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReshapeFunctor {
  ReshapeFunctor() {}

  void operator()(const Tensor *input,
                  const std::vector<index_t> &out_shape,
                  Tensor *output,
                  StatsFuture *future) {
    output->ResizeWithBuffer(out_shape, input->UnderlyingBuffer());
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_RESHAPE_H_

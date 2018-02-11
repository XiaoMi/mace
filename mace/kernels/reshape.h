//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_RESHAPE_H_
#define MACE_KERNELS_RESHAPE_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReshapeFunctor {
  ReshapeFunctor() {}

  void operator()(const Tensor *input,
                  const std::vector<index_t> &out_shape,
                  Tensor *output,
                  StatsFuture *future) {
    output->Resize(out_shape);
    output->CopyBytes(input->raw_data(), input->size() * sizeof(T));
  }
};


}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_RESHAPE_H_

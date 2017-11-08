//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONV_2D_H_
#define MACE_KERNELS_CONV_2D_H_

#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct SpaceToBatchFunctor {
  SpaceToBatchFunctor(const bool b2s = false): b2s_(b2s){}

  void operator()(Tensor *input_tensor,
                  const Tensor *block_shape_tensor,
                  const Tensor *paddings_tensor,
                  Tensor *output_tensor) {
    MACE_CHECK_NOTNULL(input_tensor);
    MACE_CHECK_NOTNULL(block_shape_tensor);
    MACE_CHECK_NOTNULL(paddings_tensor);
    MACE_CHECK_NOTNULL(output_tensor);
  }

  bool b2s_;
};

template <>
void SpaceToBatchFunctor<DeviceType::OPENCL, float>::operator()(Tensor *input_tensor,
                                                                const Tensor *block_shape_tensor,
                                                                const Tensor *paddings_tensor,
                                                                Tensor *output);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_2D_H_

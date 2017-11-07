//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/resize_bilinear.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <>
void ResizeBilinearFunctor<DeviceType::OPENCL, float>::operator()(
    const Tensor *input, const Tensor *resize_dims, Tensor *output) {}

}  // namespace kernels
}  // namespace mace

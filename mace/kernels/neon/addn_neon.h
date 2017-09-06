//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ADDN_NEON_H_
#define MACE_KERNELS_ADDN_NEON_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

void NeonAddNFuntion_float(const vector<const Tensor *> &input_tensor,
                           Tensor *output_tensor);

} // namespace kernels
} // namespace mace

#endif // MACE_KERNELS_ADDN_NEON_H_

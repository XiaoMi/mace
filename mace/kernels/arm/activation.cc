//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/activation.h"

namespace mace {
namespace kernels {

void ActivationFunctor<DeviceType::NEON, float>::operator()(
  const Tensor *input,
  const Tensor *alpha,
  Tensor *output,
  StatsFuture *future) {
  const float *input_ptr = input->data<float>();
  float *output_ptr = output->mutable_data<float>();
  if (activation_ == PRELU) {
    MACE_CHECK_NOTNULL(alpha);
    const float *alpha_ptr = alpha->data<float>();
    const index_t outer_size = output->dim(0);
    const index_t inner_size = output->dim(2) * output->dim(3);
    PReLUActivation(input_ptr, outer_size, input->dim(1), inner_size, alpha_ptr,
                    output_ptr);
  } else {
    DoActivation(input_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_);
  }
}

}  // namespace kernels
}  // namespace mace




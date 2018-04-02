//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/fully_connected.h"
#include "mace/kernels/gemm.h"

namespace mace {
namespace kernels {

void FullyConnectedFunctor<DeviceType::NEON,
                           float>::operator()(const Tensor *input,
                                              const Tensor *weight,
                                              const Tensor *bias,
                                              Tensor *output,
                                              StatsFuture *future) {
  std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
  output->Resize(output_shape);
  const index_t N = output->dim(0);
  const index_t input_size = weight->dim(1);
  const index_t output_size = weight->dim(0);
  const float *input_ptr = input->data<float>();
  const float *weight_ptr = weight->data<float>();
  const float *bias_ptr = bias == nullptr ? nullptr : bias->data<float>();
  float *output_ptr = output->mutable_data<float>();

  for (int i = 0; i < N; ++i) {
    Gemm(weight_ptr, input_ptr, 1, output_size, input_size, 1, output_ptr);
    for (int j = 0; j < output_size; ++j) {
      output_ptr[j] += bias_ptr[j];
    }
  }

  DoActivation(output_ptr, output_ptr, output->size(), activation_,
               relux_max_limit_);
}

}  // namespace kernels
}  // namespace mace

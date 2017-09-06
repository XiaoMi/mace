//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ADDN_H_
#define MACE_KERNELS_ADDN_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template<typename T>
void AddNFuntion(const vector<const Tensor*>& input_tensor, Tensor *output_tensor) {
  int n = input_tensor.size();
  CHECK(n > 1);
  CHECK_NOTNULL(input_tensor[0]);
  int64 size = input_tensor[0]->size();
  vector<const T*> inputs(n);
  for (int i = 0; i < n; ++i) {
    inputs[i] = input_tensor[i]->data<float>();
  }
  output_tensor->ResizeLike(input_tensor[0]);
  float* output = output_tensor->mutable_data<T>();

  for (int i = 0; i < n; ++i) {
    for (int64 j = 0; j < size; ++j) {
      output[j] += inputs[i][j];
    }
  }
}

} //  namespace kernels
} //  namespace mace

#endif // MACE_KERNELS_ADDN_H_
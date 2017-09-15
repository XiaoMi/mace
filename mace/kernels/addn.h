//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_ADDN_H_
#define MACE_KERNELS_ADDN_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct AddNFunctor {
  void operator()(const vector<const T*>& inputs, T* output, index_t size) {
    memset(output, 0, size * sizeof(T));
    int n = inputs.size();
    for (int i = 0; i < n; ++i) {
      for (index_t j = 0; j < size; ++j) {
        output[j] += inputs[i][j];
      }
    }
  }
};

template <>
void AddNFunctor<DeviceType::NEON, float>::operator()(
    const vector<const float*>& inputs, float* output, index_t size);

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_ADDN_H_
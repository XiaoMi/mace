//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONCAT_H_
#define MACE_KERNELS_CONCAT_H_

#include "mace/proto/mace.pb.h"
#include "mace/core/common.h"
#include "mace/core/types.h"
namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct ConcatFunctor {
  void operator()(std::vector<const T *> &input_list,
                  const index_t inner_dim,
                  const index_t *outer_dims,
                  T *output) {
    const size_t input_count = input_list.size();
    for (int inner_idx = 0; inner_idx < inner_dim; ++inner_idx) {
      for (size_t i = 0; i < input_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output, input_list[i], outer_dims[i] * sizeof(T));
          output += outer_dims[i];
          input_list[i] += outer_dims[i];
        } else {
          for (index_t k = 0; k < outer_dims[i]; ++k) {
            *output++ = *input_list[i]++;
          }
        }
      }
    }
  }
};

}  //  namepsace kernels
} //  namespace mace

#endif //  MACE_KERNELS_CONCAT_H_

// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_KERNELS_TRANSPOSE_H_
#define MACE_KERNELS_TRANSPOSE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct TransposeFunctor {
  explicit TransposeFunctor(const std::vector<int> &dims) : dims_(dims) {}

  void operator()(const Tensor *input, Tensor *output, StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const std::vector<index_t> &input_shape = input->shape();
    const std::vector<index_t> &output_shape = output->shape();
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    if (input->dim_size() == 2) {
      MACE_CHECK(dims_[0] == 1 && dims_[1] == 0, "no need transform");
      index_t stride_i = input_shape[0];
      index_t stride_j = input_shape[1];
      for (int i = 0; i < input_shape[0]; ++i) {
        for (int j = 0; j < input_shape[1]; ++j) {
          output_data[j * stride_i + i] = input_data[i * stride_j + j];
        }
      }
    } else if (input->dim_size() == 4) {
      std::vector<index_t>
        in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                  input_shape[2] * input_shape[3], input_shape[3], 1};
      std::vector<index_t>
        out_stride{output_shape[1] * output_shape[2] * output_shape[3],
                   output_shape[2] * output_shape[3], output_shape[3], 1};

      std::vector<index_t> idim(4, 0);
      std::vector<index_t> odim(4, 0);
      for (odim[0] = 0; odim[0] < output_shape[0]; ++odim[0]) {
        for (odim[1] = 0; odim[1] < output_shape[1]; ++odim[1]) {
          for (odim[2] = 0; odim[2] < output_shape[2]; ++odim[2]) {
            for (odim[3] = 0; odim[3] < output_shape[3]; ++odim[3]) {
              idim[dims_[0]] = odim[0];
              idim[dims_[1]] = odim[1];
              idim[dims_[2]] = odim[2];
              idim[dims_[3]] = odim[3];

              output_data[odim[0] * out_stride[0] + odim[1] * out_stride[1]
                + odim[2] * out_stride[2] + odim[3]] =
                input_data[idim[0] * in_stride[0] + idim[1] * in_stride[1]
                  + idim[2] * in_stride[2] + idim[3]];
            }
          }
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  std::vector<int> dims_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_TRANSPOSE_H_

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_REORGANIZE_H_
#define MACE_KERNELS_REORGANIZE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReOrganizeFunctor {
  void operator()(const Tensor *input,
                  const std::vector<index_t> &out_shape,
                  Tensor *output,
                  StatsFuture *future) {
    const bool w2c = out_shape[3] > input->dim(3);

    const index_t height = input->dim(1);
    const index_t input_width = input->dim(2);
    const index_t input_chan = input->dim(3);
    const index_t output_width = output->dim(2);
    const index_t output_chan = output->dim(3);

    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    if (w2c) {
      MACE_CHECK((out_shape[3] % input->dim(3)) == 0);
      const index_t multiplier = out_shape[3] / input->dim(3);
#pragma omp parallel for collapse(4)
      for (index_t n = 0; n < out_shape[0]; ++n) {
        for (index_t h = 0; h < out_shape[1]; ++h) {
          for (index_t w = 0; w < out_shape[2]; ++w) {
            for (index_t c = 0; c < out_shape[3]; ++c) {
              const index_t out_offset =
                  ((n * height + h) * output_width + w)
                      * output_chan + c;
              const index_t in_w_idx = w + (c % multiplier) * output_width;
              const index_t in_chan_idx = c / multiplier;
              const index_t in_offset =
                  ((n * height + h) * input_width + in_w_idx)
                      * input_chan + in_chan_idx;
              output_ptr[out_offset] = input_ptr[in_offset];
            }
          }
        }
      }
    } else {
      MACE_CHECK((input->dim(3) % out_shape[3]) == 0);
      const index_t multiplier = input->dim(3) / out_shape[3];

#pragma omp parallel for collapse(4)
      for (index_t n = 0; n < out_shape[0]; ++n) {
        for (index_t h = 0; h < out_shape[1]; ++h) {
          for (index_t w = 0; w < out_shape[2]; ++w) {
            for (index_t c = 0; c < out_shape[3]; ++c) {
              const index_t out_offset =
                  ((n * height + h) * output_width + w)
                      * output_chan + c;
              const index_t in_w_idx = w % input_width;
              const index_t in_chan_idx = w / input_width + c * multiplier;
              const index_t in_offset =
                  ((n * height + h) * input_width + in_w_idx)
                      * input_chan + in_chan_idx;
              output_ptr[out_offset] = input_ptr[in_offset];
            }
          }
        }
      }
    }
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_REORGANIZE_H_

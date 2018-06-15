//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_REDUCE_MEAN_H_
#define MACE_KERNELS_REDUCE_MEAN_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif

namespace mace {
namespace kernels {

struct ReduceFunctorBase {
  ReduceFunctorBase(const std::vector<int> &axis,
                    const bool keep_dims)
      : keep_dims_(keep_dims),
        axis_(axis) {}
  bool keep_dims_;
  bool reduce_first_axis_;
  const std::vector<int> axis_;
  std::vector<int> data_reshape_;
  std::vector<index_t> out_shape_;
};

template <DeviceType D, typename T>
struct ReduceMeanFunctor : ReduceFunctorBase{
  ReduceMeanFunctor(const std::vector<int> &axis,
                    const bool keep_dims)
      : ReduceFunctorBase(axis, keep_dims) {}

  void Simplify(const Tensor *input) {
    std::vector<bool> bitmap(static_cast<uint32_t>(input->dim_size()), false);
    if (axis_.size() == 0) {
      for (int i = 0; i < input->dim_size(); ++i) {
        bitmap[i] = true;
      }
    } else {
      for (unsigned int i = 0; i < axis_.size(); ++i) {
        const int index = axis_[i] >= 0 ?
                          axis_[i] :
                          axis_[i] + input->dim_size();
        bitmap[index] = true;
      }
    }
    out_shape_.clear();
    for (unsigned int i = 0; i < input->dim_size(); ++i) {
      if (!bitmap[i]) {
        out_shape_.push_back(input->dim(i));
      } else if (keep_dims_) {
        out_shape_.push_back(1);
      }
    }
    data_reshape_.clear();
    unsigned int dim_index = 0;
    for (; dim_index < input->dim_size(); ++dim_index) {
      if (input->dim(dim_index) != 1) break;
    }
    if (dim_index >= input->dim_size()) {
      reduce_first_axis_ = true;
    } else {
      reduce_first_axis_ = bitmap[dim_index];
      data_reshape_.push_back(input->dim(dim_index));
      ++dim_index;
      for (; dim_index < input->dim_size(); ++dim_index) {
        const int n = input->dim(dim_index);
        if (n == 1) {
          bitmap[dim_index] = bitmap[dim_index - 1];
        }
        if (bitmap[dim_index-1] != bitmap[dim_index]) {
          data_reshape_.push_back(n);
        } else {
          data_reshape_.back() *= n;
        }
      }
    }
  }

  void Compute(const Tensor *input, Tensor *output) {
    Tensor::MappingGuard input_mapper(input);
    const T *input_ptr = input->data<T>();
    Tensor::MappingGuard output_map(output);
    T *output_ptr = output->mutable_data<T>();
    memset(output_ptr, 0, output->size() * sizeof(T));
    switch (data_reshape_.size()) {
      case 1:
        if (reduce_first_axis_) {
          T sum = 0;
#pragma omp parallel for reduction(+:sum)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            sum = sum + input_ptr[i];
          }
          output_ptr[0] = sum / data_reshape_[0];
        } else {
#pragma omp parallel for
          for (int i = 0; i < data_reshape_[0]; ++i) {
            output_ptr[i] = input_ptr[i];
          }
        }
        break;
      case 2:
        if (reduce_first_axis_) {
#pragma omp parallel for
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[0]; ++j) {
              output_ptr[i] += input_ptr[j * data_reshape_[1] + i];
            }
            output_ptr[i] /= data_reshape_[0];
          }
        } else {
#pragma omp parallel for
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[1]; ++j) {
              output_ptr[i] += input_ptr[i * data_reshape_[1] + j];
            }
            output_ptr[i] /= data_reshape_[1];
          }
        }
        break;
      case 3:
        if (reduce_first_axis_) {
#pragma omp parallel for
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                output_ptr[i] +=
                    input_ptr[(k * data_reshape_[1] + i) * data_reshape_[2]
                        + j];
              }
            }
            output_ptr[i] /= (data_reshape_[0] * data_reshape_[2]);
          }
        } else {
#pragma omp parallel for collapse(2)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                output_ptr[i * data_reshape_[2] + j] +=
                    input_ptr[(i * data_reshape_[1] + k) * data_reshape_[2]
                        + j];
              }
              output_ptr[i * data_reshape_[2] + j] /= data_reshape_[1];
            }
          }
        }
        break;
      case 4:
        if (reduce_first_axis_) {
#pragma omp parallel for collapse(2)
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[3]; ++j) {
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  output_ptr[i * data_reshape_[3] + j] +=
                      input_ptr[((t * data_reshape_[1] + i) *
                          data_reshape_[2] + k)*data_reshape_[3] + j];
                }
              }
              output_ptr[i * data_reshape_[3] + j] /=
                  (data_reshape_[0] * data_reshape_[2]);
            }
          }
        } else {
#pragma omp parallel for collapse(2)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  output_ptr[i * data_reshape_[2] + j] +=
                      input_ptr[((i * data_reshape_[1] + k) *
                          data_reshape_[2] + j)*data_reshape_[3] + t];
                }
              }
              output_ptr[i * data_reshape_[2] + j] /=
                  (data_reshape_[1] * data_reshape_[3]);
            }
          }
        }
        break;
      default:
        MACE_CHECK(false, "not implemented in mace")
          << "data reshape size" << data_reshape_.size()
          << "reduce first axis:" << reduce_first_axis_;
        break;
    }
  }

  MaceStatus operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    Simplify(input);
    output->Resize(out_shape_);
    Compute(input, output);
    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct ReduceMeanFunctor<DeviceType::GPU, T>
    : ReduceFunctorBase {
  ReduceMeanFunctor(const std::vector<int> axis,
                    const bool keep_dims)
      : ReduceFunctorBase(axis, keep_dims) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output_tensor,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_REDUCE_MEAN_H_

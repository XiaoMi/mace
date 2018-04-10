//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_ELTWISE_H_
#define MACE_KERNELS_ELTWISE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

enum EltwiseType {
  PROD = 0,
  SUM = 1,
  MAX = 2,
  MIN = 3,
  SUB = 4,
};

struct EltwiseFunctorBase {
  EltwiseFunctorBase(const EltwiseType type, const std::vector<float> &coeff)
      : type_(type), coeff_(coeff) {}

  EltwiseType type_;
  std::vector<float> coeff_;
};

template <DeviceType D, typename T>
struct EltwiseFunctor : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type, const std::vector<float> &coeff)
      : EltwiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input0,
                  const Tensor *input1,
                  Tensor *output,
                  StatsFuture *future) {
    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard input1_guard(input1);
    Tensor::MappingGuard output_guard(output);

    const T *input0_ptr = input0->data<T>();
    const T *input1_ptr = input1->data<T>();
    T *output_ptr = output->mutable_data<T>();
    const index_t size = input0->size();

    switch (type_) {
      case PROD:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = input0_ptr[i] * input1_ptr[i];
        }
        break;
      case SUM:
        if (coeff_.empty()) {
#pragma omp parallel for
          for (index_t i = 0; i < size; ++i) {
            output_ptr[i] = input0_ptr[i] + input1_ptr[i];
          }
        } else {
#pragma omp parallel for
          for (index_t i = 0; i < size; ++i) {
            output_ptr[i] =
                coeff_[0] * input0_ptr[i] + coeff_[1] * input1_ptr[i];
          }
        }
        break;
      case MAX:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = std::max<T>(input0_ptr[i], input1_ptr[i]);
        }
        break;
      case MIN:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = std::min<T>(input0_ptr[i], input1_ptr[i]);
        }
        break;
      case SUB:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = input0_ptr[i] - input1_ptr[i];
        }
        break;
      default:
        LOG(FATAL) << "Eltwise op not support type " << type_;
    }
  }
};

template <typename T>
struct EltwiseFunctor<DeviceType::OPENCL, T> : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type, const std::vector<float> &coeff)
      : EltwiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input0,
                  const Tensor *input1,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ELTWISE_H_

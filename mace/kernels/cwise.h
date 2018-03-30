//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_CWISE_H_
#define MACE_KERNELS_CWISE_H_

#include <algorithm>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

enum CWiseType {
  MUL = 0,
  ADD = 1,
  MAX = 2,
  MIN = 3,
  SUB = 4,
  DIV = 5,
  NEG = 6,
  ABS = 7,
};

struct CWiseFunctorBase {
  CWiseFunctorBase(const CWiseType type, const float coeff)
      : type_(type), coeff_(coeff) {}

  CWiseType type_;
  float coeff_;
};

template <DeviceType D, typename T>
struct CWiseFunctor : CWiseFunctorBase {
  CWiseFunctor(const CWiseType type, const float coeff)
      : CWiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);

    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();
    const index_t size = input->size();

    switch (type_) {
      case MUL:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] =  coeff_ * input_ptr[i];
        }
        break;
      case ADD:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] =  coeff_ + input_ptr[i];
        }
        break;
      case MAX:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = std::max<T>(input_ptr[i], coeff_);
        }
        break;
      case MIN:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] = std::min<T>(input_ptr[i], coeff_);
        }
        break;
      case SUB:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] =  input_ptr[i] - coeff_;
        }
        break;
      case DIV:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] =  input_ptr[i] / coeff_;
        }
        break;
      case NEG:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output_ptr[i] =  0 - input_ptr[i];
        }
        break;
      case ABS:
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          T val = input_ptr[i];
          output_ptr[i] =  (val > 0)? val : 0 - val;
        }
        break;
      default:
        LOG(FATAL) << "CWise op not support type " << type_;
    }
  }
};

template <typename T>
struct CWiseFunctor<DeviceType::OPENCL, T> : CWiseFunctorBase {
  CWiseFunctor(const CWiseType type, const float coeff)
      : CWiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CWISE_H_

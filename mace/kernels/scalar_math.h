//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_SCALAR_MATH_H_
#define MACE_KERNELS_SCALAR_MATH_H_

#include <algorithm>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

enum ScalarMathType {
  MUL = 0,
  ADD = 1,
  MAX = 2,
  MIN = 3,
  SUB = 4,
  DIV = 5,
};

struct ScalarMathFunctorBase {
  ScalarMathFunctorBase(const ScalarMathType type, const float coeff)
      : type_(type), coeff_(coeff) {}

  ScalarMathType type_;
  float coeff_;
};

template <DeviceType D, typename T>
struct ScalarMathFunctor : ScalarMathFunctorBase {
  ScalarMathFunctor(const ScalarMathType type, const float coeff)
      : ScalarMathFunctorBase(type, coeff) {}

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
      default:
        LOG(FATAL) << "ScalarMath op not support type " << type_;
    }
  }
};

template <typename T>
struct ScalarMathFunctor<DeviceType::OPENCL, T> : ScalarMathFunctorBase {
  ScalarMathFunctor(const ScalarMathType type, const float coeff)
      : ScalarMathFunctorBase(type, coeff) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SCALAR_MATH_H_

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

#ifndef MACE_KERNELS_CWISE_H_
#define MACE_KERNELS_CWISE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

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

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct CWiseFunctor<DeviceType::OPENCL, T> : CWiseFunctorBase {
  CWiseFunctor(const CWiseType type, const float coeff)
      : CWiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CWISE_H_

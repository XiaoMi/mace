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
  DIV = 5,
  NEG = 6,
  ABS = 7,
  SQR_DIFF = 8,
};

struct EltwiseFunctorBase {
  EltwiseFunctorBase(const EltwiseType type,
                     const std::vector<float> &coeff)
      : type_(type), coeff_(coeff) {}

  EltwiseType type_;
  std::vector<float> coeff_;
};

template <DeviceType D, typename T>
struct EltwiseFunctor : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff)
      : EltwiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input0,
                  const Tensor *input1,
                  const index_t start_axis,
                  const bool is_scaler,
                  const float value,
                  const bool swap,
                  Tensor *output,
                  StatsFuture *future) {
    if (is_scaler) {
      Tensor::MappingGuard input0_guard(input0);
      Tensor::MappingGuard output_guard(output);

      const T *input0_ptr = input0->data<T>();
      T *output_ptr = output->mutable_data<T>();
      const index_t num = input0->size();
      switch (type_) {
        case PROD:
#pragma omp parallel for
          for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = input0_ptr[i] * value;
          }
          break;
        case SUM:
          if (coeff_.empty()) {
#pragma omp parallel for
            for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = input0_ptr[i] + value;
            }
          } else {
            const float coeff_0 = swap ? coeff_[1] : coeff_[0];
            const float coeff_1 = swap ? coeff_[0] : coeff_[1];
#pragma omp parallel for
            for (index_t i = 0; i < num; ++i) {
                output_ptr[i] = coeff_0 * input0_ptr[i] +
                        coeff_1 * value;
            }
          }
          break;
        case MAX:
#pragma omp parallel for
          for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = std::max<T>(input0_ptr[i], value);
          }
          break;
        case MIN:
#pragma omp parallel for
          for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = std::min<T>(input0_ptr[i], value);
          }
          break;
        case SUB:
#pragma omp parallel for
          for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = swap ? value - input0_ptr[i] :
                              input0_ptr[i] - value;
          }
          break;
        case DIV:
          if (!swap) {
            MACE_CHECK(fabs(value) > 1e-6, "cannot divided by 0.");
#pragma omp parallel for
            for (index_t i = 0; i < num; ++i) {
              output_ptr[i] = input0_ptr[i] / value;
            }
          } else {
#pragma omp parallel for
            for (index_t i = 0; i < num; ++i) {
              MACE_CHECK(fabs(input0_ptr[i]) > 1e-6, "cannot divided by 0.");
              output_ptr[i] = value / input0_ptr[i];
            }
          }
          break;
        case SQR_DIFF:
#pragma omp parallel for
          for (index_t i = 0; i < num; ++i) {
              const float tmp = input0_ptr[i] - value;
              output_ptr[i] = tmp * tmp;
          }
          break;
        default:
          LOG(FATAL) << "Eltwise op not support type " << type_;
      }
    } else {
      MACE_CHECK_NOTNULL(input0);
      MACE_CHECK_NOTNULL(input1);
      Tensor::MappingGuard input0_guard(input0);
      Tensor::MappingGuard input1_guard(input1);
      Tensor::MappingGuard output_guard(output);

      const T *input0_ptr = input0->data<T>();
      const T *input1_ptr = input1->data<T>();
      T *output_ptr = output->mutable_data<T>();
      const index_t size0 = input0->size();
      const index_t size1 = input1->size();

      const index_t num = size0 / size1;
      switch (type_) {
        case PROD:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j= 0; j < size1; ++j) {
              output_ptr[i * size1 + j] =
                  input0_ptr[i * size1 + j] * input1_ptr[j];
            }
          }
          break;
        case SUM:
          if (coeff_.empty()) {
#pragma omp parallel for collapse(2)
            for (index_t i = 0; i < num; ++i) {
              for (index_t j = 0; j < size1; ++j) {
                output_ptr[i * size1 + j] =
                    input0_ptr[i * size1 + j] + input1_ptr[j];
              }
            }
          } else {
            const float coeff_0 = swap ? coeff_[1] : coeff_[0];
            const float coeff_1 = swap ? coeff_[0] : coeff_[1];
#pragma omp parallel for collapse(2)
            for (index_t i = 0; i < num; ++i) {
              for (index_t j = 0; j < size1; ++j) {
                output_ptr[i * size1 + j] =
                    coeff_0 * input0_ptr[i * size1 + j] +
                        coeff_1 * input1_ptr[j];
              }
            }
          }
          break;
        case MAX:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j = 0; j < size1; ++j) {
              output_ptr[i * size1 + j] =
                  std::max<T>(input0_ptr[i * size1 + j], input1_ptr[j]);
            }
          }
          break;
        case MIN:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j = 0; j < size1; ++j) {
              output_ptr[i * size1 + j] =
                  std::min<T>(input0_ptr[i * size1 + j], input1_ptr[j]);
            }
          }
          break;
        case SUB:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j = 0; j < size1; ++j) {
              output_ptr[i * size1 + j] = swap ?
                  input0_ptr[i * size1 + j] - input1_ptr[j] :
                  input1_ptr[j] - input0_ptr[i * size1 + j];
            }
          }
          break;
        case DIV:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j = 0; j < size1; ++j) {
              if (!swap) {
                MACE_CHECK(fabs(input1_ptr[j]) > 1e-6, "cannot divided by 0.");
                output_ptr[i * size1 + j] =
                    input0_ptr[i * size1 + j] / input1_ptr[j];
              } else {
                MACE_CHECK(fabs(input0_ptr[i * size1 + j]) > 1e-6,
                           "cannot divided by 0.");
                output_ptr[i * size1 + j] =
                    input1_ptr[j] / input0_ptr[i * size1 + j];
              }
            }
          }
          break;
        case SQR_DIFF:
#pragma omp parallel for collapse(2)
          for (index_t i = 0; i < num; ++i) {
            for (index_t j = 0; j < size1; ++j) {
              const T tmp = input0_ptr[i * size1 + j] - input1_ptr[j];
              output_ptr[i * size1 + j] = tmp * tmp;
            }
          }
          break;
        default:
          LOG(FATAL) << "Eltwise op not support type " << type_;
      }
    }
  }
};

template <typename T>
struct EltwiseFunctor<DeviceType::OPENCL, T> : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff)
      : EltwiseFunctorBase(type, coeff) {}

  void operator()(const Tensor *input0,
                  const Tensor *input1,
                  const index_t start_axis,
                  const bool is_scaler,
                  const float value,
                  const bool swap,
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

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
#include <utility>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

enum EltwiseType {
  SUM = 0,
  SUB = 1,
  PROD = 2,
  DIV = 3,
  MIN = 4,
  MAX = 5,
  NEG = 6,
  ABS = 7,
  SQR_DIFF = 8,
  POW = 9,
  NONE = 10,
};

inline void TensorBroadcastEltwise(const EltwiseType type,
                                   const float *input0,
                                   const float *input1,
                                   const std::vector<float> &coeff,
                                   const index_t diff_size,
                                   const index_t common_size,
                                   const bool swapped,
                                   float *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] + input1[i];
          }
        }
      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] * coeff_copy[0] +
                input1[i] * coeff_copy[1];
          }
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] - input1[i];
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input1[i] - input0[i + d * common_size];
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(2)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input0[i + d * common_size] * input1[i];
        }
      }
      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] / input1[i];
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input1[i] / input0[i + d * common_size];
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(2)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::min(input0[i + d * common_size], input1[i]);
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(2)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::max(input0[i + d * common_size], input1[i]);
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(2)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::pow(input0[i + d * common_size] - input1[i], 2.f);
        }
      }
      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::pow(input0[i + d * common_size], input1[i]);
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::pow(input1[i], input0[i + d * common_size]);
          }
        }
      }
      break;
    case NEG:
#pragma omp parallel for
      for (index_t i = 0; i < diff_size * common_size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for
      for (index_t i = 0; i < diff_size * common_size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
inline void TensorEltwise(const EltwiseType type,
                          const float *input0,
                          const float *input1,
                          const std::vector<float> &coeff,
                          const index_t size,
                          const bool swapped,
                          float *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] + input1[i];
        }

      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] * coeff_copy[0] + input1[i] * coeff_copy[1];
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] - input1[i];
        }

      } else {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1[i] - input0[i];
        }
      }
      break;
    case PROD:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * input1[i];
      }

      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] / input1[i];
        }

      } else {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1[i] / input0[i];
        }
      }
      break;
    case MIN:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min(input0[i], input1[i]);
      }

      break;
    case MAX:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max(input0[i], input1[i]);
      }

      break;
    case SQR_DIFF:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - input1[i], 2.f);
      }

      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input0[i], input1[i]);
        }
      } else {
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input1[i], input0[i]);
        }
      }
      break;
    case NEG:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
inline void TensorScalarEltwise(const EltwiseType type,
                                const float *input0,
                                const float input1,
                                const std::vector<float> &coeff,
                                const index_t size,
                                const bool swapped,
                                float *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] + input1;
        }

      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] * coeff_copy[0] + input1 * coeff_copy[1];
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] - input1;
        }

      } else {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1 - input0[i];
        }
      }
      break;
    case PROD:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * input1;
      }

      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] / input1;
        }

      } else {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1 / input0[i];
        }
      }
      break;
    case MIN:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min(input0[i], input1);
      }

      break;
    case MAX:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max(input0[i], input1);
      }

      break;
    case SQR_DIFF:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - input1, 2.f);
      }

      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input0[i], input1);
        }
      } else {
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input1, input0[i]);
        }
      }
      break;
    case NEG:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

inline void TensorEltwisePerChannel(const EltwiseType type,
                                    const float *input0,
                                    const float *input1,
                                    const std::vector<float> &coeff,
                                    const index_t batch0,
                                    const index_t batch1,
                                    const index_t channel,
                                    const index_t image_size,
                                    const bool swapped,
                                    float *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] + in1_ptr[c];
            }
          }
        }
      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] =
                  in0_ptr[i] * coeff_copy[0] + in1_ptr[c] * coeff_copy[1];
            }
          }
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] - in1_ptr[c];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in1_ptr[c] - in0_ptr[i];
            }
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          float *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] * in1_ptr[c];
          }
        }
      }
      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] / in1_ptr[c];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in1_ptr[c] / in0_ptr[i];
            }
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          float *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::min(in0_ptr[i], in1_ptr[c]);
          }
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          float *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::max(in0_ptr[i], in1_ptr[c]);
          }
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          float *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::pow(in0_ptr[i] - in1_ptr[c], 2.f);
          }
        }
      }
      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::pow(in0_ptr[i], in1_ptr[c]);
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const float *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const float *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            float *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::pow(in1_ptr[c], in0_ptr[i]);
            }
          }
        }
      }
      break;
    case NEG:
#pragma omp parallel for
      for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for
      for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

struct EltwiseFunctorBase {
  EltwiseFunctorBase(const EltwiseType type,
                     const std::vector<float> &coeff,
                     const float value,
                     const DataFormat data_format)
      : type_(type), coeff_(coeff), value_(value), data_format_(data_format) {}

  EltwiseType type_;
  std::vector<float> coeff_;
  float value_;
  DataFormat data_format_;
};

template <DeviceType D, typename T>
struct EltwiseFunctor;

template <>
struct EltwiseFunctor<DeviceType::CPU, float> : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff,
                 const float value,
                 const DataFormat data_format)
      : EltwiseFunctorBase(type, coeff, value, data_format) {}

  MaceStatus operator()(const Tensor *input0,
                        const Tensor *input1,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    if (input1 == nullptr) {
      scalar_tensor_.Resize({});
      Tensor::MappingGuard guard(&scalar_tensor_);
      auto scalar_data = scalar_tensor_.mutable_data<float>();
      scalar_data[0] = value_;
      input1 = &scalar_tensor_;
    }

    bool swapped = false;
    if (input0->size() < input1->size()) {
      std::swap(input0, input1);
      swapped = true;
    }

    // check if we can broadcast tensor
    uint32_t rank_diff =
        static_cast<uint32_t>(input0->dim_size() - input1->dim_size());
    if (data_format_ == NCHW) {
      MACE_CHECK(
          (input0->dim_size() == 4)
              && ((input1->dim_size() == 0)
                  || (input1->dim_size() == 4
                      && input1->dim(1) == input0->dim(1)
                      && (input1->dim(0) == input0->dim(0)
                          || input1->dim(0) == 1))
                  || (input1->dim_size() == 1
                      && input1->dim(0) == input0->dim(1))),
          "only support broadcast channel dimension");
    } else {
      if (rank_diff > 0 && rank_diff < input0->dim_size()) {
        for (uint32_t i = 0; i < input1->dim_size(); ++i) {
          MACE_CHECK(input0->dim(rank_diff + i) == input1->dim(i),
                     "Element-Wise op only support tail dimensions broadcast");
        }
      }
    }

    index_t common_size = input1->size();
    index_t diff_size = input0->size() / common_size;

    MACE_RETURN_IF_ERROR(output->ResizeLike(input0));

    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard input1_guard(input1);
    Tensor::MappingGuard output_guard(output);

    const float *input0_ptr = input0->data<float>();
    const float *input1_ptr = input1->data<float>();
    float *output_ptr = output->mutable_data<float>();

    if (data_format_ == NCHW && input1->dim_size() > 0 &&
        input1->size() < input0->size()) {
      TensorEltwisePerChannel(
          type_, input0_ptr, input1_ptr, coeff_, input0->dim(0),
          input1->dim_size() == 1 ? 1 : input1->dim(0), input0->dim(1),
          input0->dim(2) * input0->dim(3), swapped, output_ptr);

    } else {
      if (input1->size() == input0->size()) {
        TensorEltwise(type_, input0_ptr, input1_ptr, coeff_, input0->size(),
                      swapped, output_ptr);
      } else if (input1->size() < input0->size()) {
        if (input1->size() > 1) {
          TensorBroadcastEltwise(type_, input0_ptr, input1_ptr, coeff_,
                                 diff_size, common_size, swapped, output_ptr);
        } else {
          TensorScalarEltwise(type_, input0_ptr, input1_ptr[0], coeff_,
                              input0->size(), swapped, output_ptr);
        }
      }
    }

    return MACE_SUCCESS;
  }

  Tensor scalar_tensor_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct EltwiseFunctor<DeviceType::GPU, T> : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff,
                 const float value,
                 const DataFormat data_format)
      : EltwiseFunctorBase(type, coeff, value, data_format) {}

  MaceStatus operator()(const Tensor *input0,
                        const Tensor *input1,
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

#endif  // MACE_KERNELS_ELTWISE_H_

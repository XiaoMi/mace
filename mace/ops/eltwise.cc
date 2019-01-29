// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/eltwise.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/utils/quantize.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/eltwise.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {


inline index_t GetIndex(const std::vector<index_t> &shape,
                        const std::vector<index_t> &index) {
  index_t idx = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1) {
      idx = idx * shape[i] + index[i];
    }
  }
  return idx;
}

inline void IncreaseIndex(const std::vector<index_t> &shape,
                          std::vector<index_t> *index) {
  for (index_t i = static_cast<index_t>(shape.size()) - 1; i >= 0; --i) {
    ++(*index)[i];
    if ((*index)[i] >= shape[i]) {
      (*index)[i] -= shape[i];
    } else {
      break;
    }
  }
}

template <typename T, typename DstType>
inline void TensorGeneralBroadcastEltwise(
    const EltwiseType type,
    const T *input0,
    const T *input1,
    const std::vector<float> &coeff,
    const bool swapped,
    const std::vector<index_t> &input0_shape,
    const std::vector<index_t> &input1_shape,
    const std::vector<index_t> &output_shape,
    DstType *output) {
  const index_t output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<index_t>());
  std::vector<index_t> out_index(output_shape.size(), 0);
  switch (type) {
    case SUM:
      if (coeff.empty()) {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = input0[idx0] + input1[idx1];
          IncreaseIndex(output_shape, &out_index);
        }
      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] =
              input0[idx0] * coeff_copy[0] + input1[idx1] * coeff_copy[1];
          IncreaseIndex(output_shape, &out_index);
        }
      }
      break;
    case SUB:
      if (!swapped) {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = input0[idx0] - input1[idx1];
          IncreaseIndex(output_shape, &out_index);
        }
      } else {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = input1[idx1] - input0[idx0];
          IncreaseIndex(output_shape, &out_index);
        }
      }
      break;
    case PROD:
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input0[idx0] * input1[idx1];
        IncreaseIndex(output_shape, &out_index);
      }
      break;
    case DIV:
      if (!swapped) {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = input0[idx0] / input1[idx1];
          IncreaseIndex(output_shape, &out_index);
        }
      } else {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = input1[idx1] / input0[idx0];
          IncreaseIndex(output_shape, &out_index);
        }
      }
      break;
    case FLOOR_DIV:
      if (!swapped) {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = std::floor(input0[idx0] / input1[idx1]);
          IncreaseIndex(output_shape, &out_index);
        }
      } else {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = std::floor(input1[idx1] / input0[idx0]);
          IncreaseIndex(output_shape, &out_index);
        }
      }
      break;
    case MIN:
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::min(input1[idx1], input0[idx0]);
        IncreaseIndex(output_shape, &out_index);
      }
      break;
    case MAX:
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::max(input1[idx1], input0[idx0]);
        IncreaseIndex(output_shape, &out_index);
      }
      break;
    case SQR_DIFF:
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::pow(input1[idx1] - input0[idx0], 2.f);
        IncreaseIndex(output_shape, &out_index);
      }
      break;
    case POW:
      if (!swapped) {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = std::pow(input0[idx0], input1[idx1]);
          IncreaseIndex(output_shape, &out_index);
        }
      } else {
        for (index_t i = 0; i < output_size; ++i) {
          const index_t idx0 = GetIndex(input0_shape, out_index);
          const index_t idx1 = GetIndex(input1_shape, out_index);
          output[i] = std::pow(input1[idx1], input0[idx0]);
          IncreaseIndex(output_shape, &out_index);
        }
      }
      break;
    case EQUAL:
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input1[idx1] == input0[idx0];
        IncreaseIndex(output_shape, &out_index);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <typename T, typename DstType>
inline void TensorBroadcastEltwise(const EltwiseType type,
                                   const T *input0,
                                   const T *input1,
                                   const std::vector<float> &coeff,
                                   const index_t diff_size,
                                   const index_t common_size,
                                   const bool swapped,
                                   DstType *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for collapse(2) schedule(runtime)
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
#pragma omp parallel for collapse(2) schedule(runtime)
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
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] - input1[i];
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input1[i] - input0[i + d * common_size];
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input0[i + d * common_size] * input1[i];
        }
      }
      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] / input1[i];
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input1[i] / input0[i + d * common_size];
          }
        }
      }
      break;
    case FLOOR_DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::floor(input0[i + d * common_size] / input1[i]);
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::floor(input1[i] / input0[i + d * common_size]);
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::min(input0[i + d * common_size], input1[i]);
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::max(input0[i + d * common_size], input1[i]);
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::pow(input0[i + d * common_size] - input1[i], 2.f);
        }
      }
      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::pow(input0[i + d * common_size], input1[i]);
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t d = 0; d < diff_size; ++d) {
          for (index_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                std::pow(input1[i], input0[i + d * common_size]);
          }
        }
      }
      break;
    case NEG:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < diff_size * common_size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < diff_size * common_size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    case EQUAL:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              input0[i + d * common_size] == input1[i];
        }
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
template <typename T, typename DstType>
inline void TensorEltwise(const EltwiseType type,
                          const T *input0,
                          const T *input1,
                          const std::vector<float> &coeff,
                          const index_t size,
                          const bool swapped,
                          DstType *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] + input1[i];
        }

      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] * coeff_copy[0] + input1[i] * coeff_copy[1];
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] - input1[i];
        }

      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1[i] - input0[i];
        }
      }
      break;
    case PROD:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * input1[i];
      }

      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] / input1[i];
        }

      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1[i] / input0[i];
        }
      }
      break;
    case FLOOR_DIV:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::floor(input0[i] / input1[i]);
        }
      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::floor(input1[i] / input0[i]);
        }
      }
      break;
    case MIN:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min(input0[i], input1[i]);
      }

      break;
    case MAX:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max(input0[i], input1[i]);
      }

      break;
    case SQR_DIFF:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - input1[i], 2.f);
      }

      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input0[i], input1[i]);
        }
      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input1[i], input0[i]);
        }
      }
      break;
    case NEG:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    case EQUAL:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] == input1[i];
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
template <typename T, typename DstType>
inline void TensorScalarEltwise(const EltwiseType type,
                                const T *input0,
                                const T input1,
                                const std::vector<float> &coeff,
                                const index_t size,
                                const bool swapped,
                                DstType *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] + input1;
        }

      } else {
        std::vector<float> coeff_copy = coeff;
        if (swapped) {
          std::swap(coeff_copy[0], coeff_copy[1]);
        }
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] * coeff_copy[0] + input1 * coeff_copy[1];
        }
      }
      break;
    case SUB:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] - input1;
        }

      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1 - input0[i];
        }
      }
      break;
    case PROD:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * input1;
      }

      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input0[i] / input1;
        }

      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = input1 / input0[i];
        }
      }
      break;
    case FLOOR_DIV:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::floor(input0[i] / input1);
        }
      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::floor(input1 / input0[i]);
        }
      }
      break;
    case MIN:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min(input0[i], input1);
      }

      break;
    case MAX:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max(input0[i], input1);
      }

      break;
    case SQR_DIFF:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - input1, 2.f);
      }

      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input0[i], input1);
        }
      } else {
#pragma omp parallel for schedule(runtime)
        for (index_t i = 0; i < size; ++i) {
          output[i] = std::pow(input1, input0[i]);
        }
      }
      break;
    case NEG:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    case EQUAL:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] == input1;
      }

      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <typename T, typename DstType>
inline void TensorEltwisePerChannel(const EltwiseType type,
                                    const T *input0,
                                    const T *input1,
                                    const std::vector<float> &coeff,
                                    const index_t batch0,
                                    const index_t batch1,
                                    const index_t channel,
                                    const index_t image_size,
                                    const bool swapped,
                                    DstType *output) {
  switch (type) {
    case SUM:
      if (coeff.empty()) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
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
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
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
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] - in1_ptr[c];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in1_ptr[c] - in0_ptr[i];
            }
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] * in1_ptr[c];
          }
        }
      }
      break;
    case DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] / in1_ptr[c];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in1_ptr[c] / in0_ptr[i];
            }
          }
        }
      }
      break;
    case FLOOR_DIV:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::floor(in0_ptr[i] / in1_ptr[c]);
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::floor(in1_ptr[c] / in0_ptr[i]);
            }
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::min(in0_ptr[i], in1_ptr[c]);
          }
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::max(in0_ptr[i], in1_ptr[c]);
          }
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::pow(in0_ptr[i] - in1_ptr[c], 2.f);
          }
        }
      }
      break;
    case POW:
      if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::pow(in0_ptr[i], in1_ptr[c]);
            }
          }
        }
      } else {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (index_t b = 0; b < batch0; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = std::pow(in1_ptr[c], in0_ptr[i]);
            }
          }
        }
      }
      break;
    case NEG:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
        output[i] = -input0[i];
      }
      break;
    case ABS:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
        output[i] = std::fabs(input0[i]);
      }
      break;
    case EQUAL:
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] == in1_ptr[c];
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <DeviceType D, class T>
class EltwiseOp : public Operation {
 public:
  explicit EltwiseOp(OpConstructContext *context)
      : Operation(context),
        type_(static_cast<ops::EltwiseType>(Operation::GetOptionalArg<int>(
            "type", static_cast<int>(ops::EltwiseType::NONE)))),
        coeff_(Operation::GetRepeatedArgs<float>("coeff")),
        scalar_input_(Operation::GetOptionalArg<float>("scalar_input", 1.0)),
        scalar_input_index_(Operation::GetOptionalArg<int32_t>(
            "scalar_input_index", 1)),
        data_format_(static_cast<DataFormat>(Operation::GetOptionalArg<int>(
            "data_format", 0))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    if (input1 == nullptr) {
      scalar_tensor_.Resize({});
      Tensor::MappingGuard guard(&scalar_tensor_);
      auto scalar_data = scalar_tensor_.mutable_data<T>();
      scalar_data[0] = static_cast<T>(scalar_input_);
      input1 = &scalar_tensor_;
    }

    if (IsLogicalType(type_)) {
      // as we do not have bool-type tensor, we use int type
      return DoEltwise<int32_t>(input0, input1, output);
    } else {
      return DoEltwise<T>(input0, input1, output);
    }
  }

 private:
  template <typename DstType>
  MaceStatus DoEltwise(const Tensor *input0,
                       const Tensor *input1,
                       Tensor *output) {
    bool swapped = false;
    if (input0->size() < input1->size()) {
      std::swap(input0, input1);
      swapped = true;
    }
    if (scalar_input_index_ == 0) {
      swapped = !swapped;
    }

    // check if we can broadcast tensor
    uint32_t rank_diff =
        static_cast<uint32_t>(input0->dim_size() - input1->dim_size());
    if (data_format_ == NCHW) {
      MACE_CHECK(
          (input0->dim_size() == 4) &&
              ((input1->dim_size() == 0) ||
                  (input1->dim_size() == 4 &&
                      input1->dim(1) == input0->dim(1) &&
                      (input1->dim(0) == input0->dim(0) ||
                          input1->dim(0) == 1)) ||
                  (input1->dim_size() == 1 &&
                      input1->dim(0) == input0->dim(1))),
          "only support broadcast channel dimension");
    } else {
      for (uint32_t i = 0; i < input1->dim_size(); ++i) {
        MACE_CHECK(input0->dim(rank_diff + i) == 1 || input1->dim(i) == 1 ||
            input0->dim(rank_diff + i) == input1->dim(i),
                   "Element-Wise op only support tail dimensions broadcast");
      }
    }

    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard input1_guard(input1);

    const T *input0_ptr = input0->data<T>();
    const T *input1_ptr = input1->data<T>();

    if (data_format_ == NCHW && input1->dim_size() > 0) {
      MACE_RETURN_IF_ERROR(output->ResizeLike(input0));
      Tensor::MappingGuard output_guard(output);
      DstType *output_ptr = output->mutable_data<DstType>();
      if (input1->size() < input0->size()) {
        TensorEltwisePerChannel(
            type_, input0_ptr, input1_ptr, coeff_, input0->dim(0),
            input1->dim_size() == 1 ? 1 : input1->dim(0), input0->dim(1),
            input0->dim(2) * input0->dim(3), swapped, output_ptr);
      } else {
        TensorEltwise(type_, input0_ptr, input1_ptr, coeff_, input0->size(),
                      swapped, output_ptr);
      }
    } else {
      const std::vector<index_t> &input0_shape = input0->shape();
      std::vector<index_t> input1_shape(rank_diff, 1);
      input1_shape.insert(input1_shape.end(), input1->shape().begin(),
                          input1->shape().end());

      std::vector<index_t> output_shape(input0->dim_size(), 0);
      for (unsigned int i = 0; i < input0_shape.size(); ++i) {
        output_shape[i] = std::max(input0_shape[i], input1_shape[i]);
      }
      MACE_RETURN_IF_ERROR(output->Resize(output_shape));
      Tensor::MappingGuard output_guard(output);
      DstType *output_ptr = output->mutable_data<DstType>();

      bool need_general_broadcast = false;
      for (uint32_t i = 0; i < input1->dim_size(); ++i) {
        if ((input0->dim(rank_diff + i) == 1 && input1->dim(i) > 1) ||
            (input0->dim(rank_diff + i) > 1 && input1->dim(i) == 1)) {
          need_general_broadcast = true;
          break;
        }
      }

      if (input1->size() == 1) {
        TensorScalarEltwise(type_, input0_ptr, input1_ptr[0], coeff_,
                            input0->size(), swapped, output_ptr);
      } else if (input0_shape == input1_shape) {
        TensorEltwise(type_, input0_ptr, input1_ptr, coeff_, input0->size(),
                      swapped, output_ptr);
      } else if (need_general_broadcast) {
        TensorGeneralBroadcastEltwise(type_, input0_ptr, input1_ptr, coeff_,
                                      swapped, input0_shape, input1_shape,
                                      output_shape, output_ptr);
      } else {
        index_t common_size = input1->size();
        index_t diff_size = input0->size() / common_size;
        TensorBroadcastEltwise(type_, input0_ptr, input1_ptr, coeff_,
                               diff_size, common_size, swapped, output_ptr);
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  DataFormat data_format_;
  Tensor scalar_tensor_;
};

#ifdef MACE_ENABLE_QUANTIZE
template <>
class EltwiseOp<DeviceType::CPU, uint8_t> : public Operation {
 public:
  explicit EltwiseOp(OpConstructContext *context)
      : Operation(context),
        type_(static_cast<ops::EltwiseType>(Operation::GetOptionalArg<int>(
            "type", static_cast<int>(ops::EltwiseType::NONE)))),
        coeff_(Operation::GetRepeatedArgs<float>("coeff")),
        scalar_input_(Operation::GetOptionalArg<float>("scalar_input", 1.0)),
        scalar_input_index_(Operation::GetOptionalArg<int32_t>(
            "scalar_input_index", 1)),
        data_format_(static_cast<DataFormat>(Operation::GetOptionalArg<int>(
            "data_format", 0))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    MACE_CHECK(type_ == SUM, "Only support Elementwise SUM now. ");
    MACE_CHECK(input0->size() == input1->size(),
               "input0 and input1 must have the same shape.");
    MACE_CHECK(output->scale() != 0);
    MACE_RETURN_IF_ERROR(output->Resize(input0->shape()));

    constexpr int left_shift = 20;
    const double doubled_scale = 2 * std::max(input0->scale(), input1->scale());
    const double adjusted_input0_scale = input0->scale() / doubled_scale;
    const double adjusted_input1_scale = input1->scale() / doubled_scale;
    const double adjusted_output_scale =
        doubled_scale / ((1 << left_shift) * output->scale());

    int32_t input0_multiplier;
    int32_t input1_multiplier;
    int32_t output_multiplier;
    int32_t input0_shift;
    int32_t input1_shift;
    int32_t output_shift;
    QuantizeMultiplier(adjusted_input0_scale,
                       &input0_multiplier,
                       &input0_shift);
    QuantizeMultiplier(adjusted_input1_scale,
                       &input1_multiplier,
                       &input1_shift);
    QuantizeMultiplier(adjusted_output_scale,
                       &output_multiplier,
                       &output_shift);

    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard input1_guard(input1);
    Tensor::MappingGuard output_guard(output);

    auto input0_ptr = input0->data<uint8_t>();
    auto input1_ptr = input1->data<uint8_t>();
    auto output_ptr = output->mutable_data<uint8_t>();

    index_t handled_output_size = 0;
#ifdef MACE_ENABLE_NEON
    #pragma omp parallel for schedule(runtime)
    for (index_t i = handled_output_size; i <= output->size() - 8; i += 8) {
      const auto input0_val = vld1_u8(input0_ptr + i);
      const auto input1_val = vld1_u8(input1_ptr + i);
      const auto input0_val_s16 =
          vreinterpretq_s16_u16(vmovl_u8(input0_val));
      const auto input1_val_s16 =
          vreinterpretq_s16_u16(vmovl_u8(input1_val));
      const auto offset_input0 =
          vaddq_s16(input0_val_s16, vdupq_n_s16(-input0->zero_point()));
      const auto offset_input1 =
          vaddq_s16(input1_val_s16, vdupq_n_s16(-input1->zero_point()));
      auto input0_low_s32 = vmovl_s16(vget_low_s16(offset_input0));
      auto input0_high_s32 = vmovl_s16(vget_high_s16(offset_input0));
      auto input1_low_s32 = vmovl_s16(vget_low_s16(offset_input1));
      auto input1_high_s32 = vmovl_s16(vget_high_s16(offset_input1));
      const auto left_shift_dup = vdupq_n_s32(left_shift);
      input0_low_s32 = vshlq_s32(input0_low_s32, left_shift_dup);
      input0_high_s32 = vshlq_s32(input0_high_s32, left_shift_dup);
      input1_low_s32 = vshlq_s32(input1_low_s32, left_shift_dup);
      input1_high_s32 = vshlq_s32(input1_high_s32, left_shift_dup);
      input0_low_s32 = vqrdmulhq_n_s32(input0_low_s32, input0_multiplier);
      input0_high_s32 = vqrdmulhq_n_s32(input0_high_s32, input0_multiplier);
      input1_low_s32 = vqrdmulhq_n_s32(input1_low_s32, input1_multiplier);
      input1_high_s32 = vqrdmulhq_n_s32(input1_high_s32, input1_multiplier);
      const auto input0_shift_dup = vdupq_n_s32(input0_shift);
      const auto input1_shift_dup = vdupq_n_s32(input1_shift);
      input0_low_s32 = vshlq_s32(input0_low_s32, input0_shift_dup);
      input0_high_s32 = vshlq_s32(input0_high_s32, input0_shift_dup);
      input1_low_s32 = vshlq_s32(input1_low_s32, input1_shift_dup);
      input1_high_s32 = vshlq_s32(input1_high_s32, input1_shift_dup);
      auto sum_low = vaddq_s32(input0_low_s32, input1_low_s32);
      auto sum_high = vaddq_s32(input0_high_s32, input1_high_s32);
      sum_low = vqrdmulhq_n_s32(sum_low, output_multiplier);
      sum_high = vqrdmulhq_n_s32(sum_high, output_multiplier);
      sum_low = gemmlowp::RoundingDivideByPOT(sum_low, -output_shift);
      sum_high = gemmlowp::RoundingDivideByPOT(sum_high, -output_shift);
      const auto sum_low_s16 = vmovn_s32(sum_low);
      const auto sum_high_s16 = vmovn_s32(sum_high);
      const auto output_val = vaddq_s16(vcombine_s16(sum_low_s16,
                                                     sum_high_s16),
                                        vdupq_n_s16(output->zero_point()));
      vst1_u8(output_ptr + i, vqmovun_s16(output_val));
    }
    handled_output_size = output->size() - output->size() % 8;
#endif  // NEON
#pragma omp parallel for schedule(runtime)
    for (index_t i = handled_output_size; i < output->size(); ++i) {
      const int32_t offset_input0 = input0_ptr[i] - input0->zero_point();
      const int32_t offset_input1 = input1_ptr[i] - input1->zero_point();
      const int32_t shifted_input0 = offset_input0 * (1 << left_shift);
      const int32_t shifted_input1 = offset_input1 * (1 << left_shift);
      const int32_t multiplied_input0 =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input0,
                                                          input0_multiplier),
              -input0_shift);
      const int32_t multiplied_input1 =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input1,
                                                          input1_multiplier),
              -input1_shift);
      const int32_t sum = multiplied_input0 + multiplied_input1;
      const int32_t output_val =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(sum,
                                                          output_multiplier),
              -output_shift) + output->zero_point();
      output_ptr[i] = Saturate<uint8_t>(output_val);
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  DataFormat data_format_;
  Tensor scalar_tensor_;
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class EltwiseOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit EltwiseOp(OpConstructContext *context)
      : Operation(context) {
    EltwiseType type = static_cast<ops::EltwiseType>(
        Operation::GetOptionalArg<int>(
            "type", static_cast<int>(ops::EltwiseType::NONE)));
    std::vector<float> coeff = Operation::GetRepeatedArgs<float>("coeff");
    float scalar_input = Operation::GetOptionalArg<float>("scalar_input", 1.0);
    int32_t scalar_input_index = Operation::GetOptionalArg<int32_t>(
            "scalar_input_index", 1);
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_.reset(new opencl::image::EltwiseKernel<T>(
          type, coeff, scalar_input, scalar_input_index));
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filters
    int input_size = operator_def_->input_size();
    Workspace *ws = context->workspace();
    for (int i = 0; i < input_size; ++i) {
      if (ws->HasTensor(operator_def_->input(i)) &&
          ws->GetTensor(operator_def_->input(i))->is_weight()) {
        MACE_CHECK(TransformFilter<T>(
            context,
            operator_def_.get(),
            i,
            OpenCLBufferType::ARGUMENT,
            mem_type) == MaceStatus::MACE_SUCCESS);
      }
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input0, input1, output);
  }

 private:
  std::unique_ptr<OpenCLEltwiseKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterEltwise(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Eltwise", EltwiseOp,
                   DeviceType::CPU, float);

  MACE_REGISTER_OP(op_registry, "Eltwise", EltwiseOp,
                   DeviceType::CPU, int32_t);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Eltwise", EltwiseOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Eltwise", EltwiseOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Eltwise", EltwiseOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

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

#ifdef MACE_ENABLE_NEON
#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/arm/q8/eltwise.h"
#endif  // MACE_ENABLE_QUANTIZE
#endif  // MACE_ENABLE_NEON

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
#include "mace/utils/memory.h"
#include "mace/core/quantize.h"
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

template<typename T, typename DstType>
inline void TensorGeneralBroadcastEltwise(
    const OpContext *context,
    const EltwiseType type,
    const T *input0,
    const T *input1,
    const std::vector<float> &coeff,
    const bool swapped,
    const std::vector<index_t> &input0_shape,
    const std::vector<index_t> &input1_shape,
    const std::vector<index_t> &output_shape,
    DstType *output) {
  MACE_UNUSED(context);

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
    default:LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template<typename T, typename DstType>
inline void TensorBroadcastEltwise(const OpContext *context,
                                   const EltwiseType type,
                                   const T *input0,
                                   const T *input1,
                                   const std::vector<float> &coeff,
                                   const index_t diff_size,
                                   const index_t common_size,
                                   const bool swapped,
                                   DstType *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    switch (type) {
      case SUM:
        if (coeff.empty()) {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input0[i + d * common_size] + input1[i];
            }
          }
        } else {
          std::vector<float> coeff_copy = coeff;
          if (swapped) {
            std::swap(coeff_copy[0], coeff_copy[1]);
          }
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input0[i + d * common_size] * coeff_copy[0] +
                      input1[i] * coeff_copy[1];
            }
          }
        }
        break;
      case SUB:
        if (!swapped) {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input0[i + d * common_size] - input1[i];
            }
          }
        } else {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input1[i] - input0[i + d * common_size];
            }
          }
        }
        break;
      case PROD:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                input0[i + d * common_size] * input1[i];
          }
        }
        break;
      case DIV:
        if (!swapped) {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input0[i + d * common_size] / input1[i];
            }
          }
        } else {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  input1[i] / input0[i + d * common_size];
            }
          }
        }
        break;
      case FLOOR_DIV:
        if (!swapped) {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  std::floor(input0[i + d * common_size] / input1[i]);
            }
          }
        } else {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  std::floor(input1[i] / input0[i + d * common_size]);
            }
          }
        }
        break;
      case MIN:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                std::min(input0[i + d * common_size], input1[i]);
          }
        }
        break;
      case MAX:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                std::max(input0[i + d * common_size], input1[i]);
          }
        }
        break;
      case SQR_DIFF:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                std::pow(input0[i + d * common_size] - input1[i], 2.f);
          }
        }
        break;
      case POW:
        if (!swapped) {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  std::pow(input0[i + d * common_size], input1[i]);
            }
          }
        } else {
          for (index_t d = start0; d < end0; d += step0) {
            for (index_t i = start1; i < end1; i += step1) {
              output[i + d * common_size] =
                  std::pow(input1[i], input0[i + d * common_size]);
            }
          }
        }
        break;
      case NEG:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] = -input0[i + d * common_size];
          }
        }
        break;
      case ABS:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                std::fabs(input0[i + d * common_size]);
          }
        }
        break;
      case EQUAL:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                input0[i + d * common_size] == input1[i];
          }
        }
        break;
      case CLIP:
        for (index_t d = start0; d < end0; d += step0) {
          for (index_t i = start1; i < end1; i += step1) {
            output[i + d * common_size] =
                std::fmaxf(coeff[0],
                           std::fminf(coeff[1], input0[i + d * common_size]));
          }
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type " << type;
    }
  }, 0, diff_size, 1, 0, common_size, 1);
}

// Multiplication is costly, so we specialize the following case.
template<typename T, typename DstType>
inline void TensorEltwise(const OpContext *context,
                          const EltwiseType type,
                          const T *input0,
                          const T *input1,
                          const std::vector<float> &coeff,
                          const index_t size,
                          const bool swapped,
                          DstType *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
    switch (type) {
      case SUM:
        if (coeff.empty()) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] + input1[i];
          }

        } else {
          std::vector<float> coeff_copy = coeff;
          if (swapped) {
            std::swap(coeff_copy[0], coeff_copy[1]);
          }
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] * coeff_copy[0] + input1[i] * coeff_copy[1];
          }
        }
        break;
      case SUB:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] - input1[i];
          }

        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = input1[i] - input0[i];
          }
        }
        break;
      case PROD:
        for (index_t i = start; i < end; i += step) {
          output[i] = input0[i] * input1[i];
        }

        break;
      case DIV:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] / input1[i];
          }

        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = input1[i] / input0[i];
          }
        }
        break;
      case FLOOR_DIV:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::floor(input0[i] / input1[i]);
          }
        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::floor(input1[i] / input0[i]);
          }
        }
        break;
      case MIN:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::min(input0[i], input1[i]);
        }

        break;
      case MAX:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::max(input0[i], input1[i]);
        }

        break;
      case SQR_DIFF:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::pow(input0[i] - input1[i], 2.f);
        }

        break;
      case POW:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::pow(input0[i], input1[i]);
          }
        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::pow(input1[i], input0[i]);
          }
        }
        break;
      case NEG:
        for (index_t i = start; i < end; i += step) {
          output[i] = -input0[i];
        }
        break;
      case ABS:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::fabs(input0[i]);
        }
        break;
      case EQUAL:
        for (index_t i = start; i < end; i += step) {
          output[i] = input0[i] == input1[i];
        }
        break;
      case CLIP:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::fmaxf(coeff[0], std::fminf(coeff[1], input0[i]));
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type " << type;
    }
  }, 0, size, 1);
}

// Multiplication is costly, so we specialize the following case.
template<typename T, typename DstType>
inline void TensorScalarEltwise(const OpContext *context,
                                const EltwiseType type,
                                const T *input0,
                                const T input1,
                                const std::vector<float> &coeff,
                                const index_t size,
                                const bool swapped,
                                DstType *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
    switch (type) {
      case SUM:
        if (coeff.empty()) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] + input1;
          }

        } else {
          std::vector<float> coeff_copy = coeff;
          if (swapped) {
            std::swap(coeff_copy[0], coeff_copy[1]);
          }
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] * coeff_copy[0] + input1 * coeff_copy[1];
          }
        }
        break;
      case SUB:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] - input1;
          }

        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = input1 - input0[i];
          }
        }
        break;
      case PROD:
        for (index_t i = start; i < end; i += step) {
          output[i] = input0[i] * input1;
        }

        break;
      case DIV:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = input0[i] / input1;
          }

        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = input1 / input0[i];
          }
        }
        break;
      case FLOOR_DIV:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::floor(input0[i] / input1);
          }
        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::floor(input1 / input0[i]);
          }
        }
        break;
      case MIN:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::min(input0[i], input1);
        }

        break;
      case MAX:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::max(input0[i], input1);
        }

        break;
      case SQR_DIFF:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::pow(input0[i] - input1, 2.f);
        }

        break;
      case POW:
        if (!swapped) {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::pow(input0[i], input1);
          }
        } else {
          for (index_t i = start; i < end; i += step) {
            output[i] = std::pow(input1, input0[i]);
          }
        }
        break;
      case NEG:
        for (index_t i = start; i < end; i += step) {
          output[i] = -input0[i];
        }
        break;
      case ABS:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::fabs(input0[i]);
        }
        break;
      case EQUAL:
        for (index_t i = start; i < end; i += step) {
          output[i] = input0[i] == input1;
        }
        break;
      case CLIP:
        for (index_t i = start; i < end; i += step) {
          output[i] = std::fmaxf(coeff[0], std::fminf(coeff[1], input0[i]));
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type " << type;
    }
  }, 0, size, 1);
}

template<typename T, typename DstType>
inline void TensorEltwisePerChannel(const OpContext *context,
                                    const EltwiseType type,
                                    const T *input0,
                                    const T *input1,
                                    const std::vector<float> &coeff,
                                    const index_t batch0,
                                    const index_t batch1,
                                    const index_t channel,
                                    const index_t image_size,
                                    const bool swapped,
                                    DstType *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    switch (type) {
      case SUM:
        if (coeff.empty()) {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (index_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in0_ptr[i] - in1_ptr[c];
              }
            }
          }
        } else {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
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
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (index_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in0_ptr[i] / in1_ptr[c];
              }
            }
          }
        } else {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (index_t i = 0; i < image_size; ++i) {
                out_ptr[i] = std::floor(in0_ptr[i] / in1_ptr[c]);
              }
            }
          }
        } else {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
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
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
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
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
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
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (index_t i = 0; i < image_size; ++i) {
                out_ptr[i] = std::pow(in0_ptr[i], in1_ptr[c]);
              }
            }
          }
        } else {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
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
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = -input0[i];
            }
          }
        }
        break;
      case ABS:
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
            for (index_t i = 0; i < image_size; ++i) {
              output[i] = std::fabs(input0[i]);
            }
          }
        }
        break;
      case EQUAL:
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (index_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] == in1_ptr[c];
            }
          }
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type " << type;
    }
  }, 0, batch0, 1, 0, channel, 1);
}

template<DeviceType D, class T>
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
        has_data_format_(Operation::GetOptionalArg<int>(
            "has_data_format", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(this->InputSize() < 3,
               "Element-Wise does not support 3 or higher inputs,"
               " you could change your model to multiple Element-Wise");
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

    if (type_ == CLIP) {
      MACE_CHECK(coeff_.size() == 2 && coeff_[0] < coeff_[1],
                 "Clip's min/max values are not correct.");
    }

    if (IsLogicalType(type_)) {
      // as we do not have bool-type tensor, we use int type
      return DoEltwise<int32_t>(context, input0, input1, output);
    } else {
      return DoEltwise<T>(context, input0, input1, output);
    }
  }

 private:
  template<typename DstType>
  MaceStatus DoEltwise(const OpContext *context,
                       const Tensor *input0,
                       const Tensor *input1,
                       Tensor *output) {
    bool swapped = false;
    if (input0->dim_size() < input1->dim_size()
        || (input0->dim_size() == input1->dim_size()
            && input0->size() < input1->size())) {
      std::swap(input0, input1);
      swapped = true;
    }
    if (scalar_input_index_ == 0) {
      swapped = !swapped;
    }

    // check if we can broadcast tensor
    uint32_t rank_diff =
        static_cast<uint32_t>(input0->dim_size() - input1->dim_size());
    if (has_data_format_) {
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

    if (has_data_format_ && input1->dim_size() > 0) {
      MACE_RETURN_IF_ERROR(output->ResizeLike(input0));
      Tensor::MappingGuard output_guard(output);
      DstType *output_ptr = output->mutable_data<DstType>();
      if (input1->size() < input0->size()) {
        TensorEltwisePerChannel(context,
                                type_,
                                input0_ptr,
                                input1_ptr,
                                coeff_,
                                input0->dim(0),
                                input1->dim_size() == 1 ? 1 : input1->dim(0),
                                input0->dim(1),
                                input0->dim(2) * input0->dim(3),
                                swapped,
                                output_ptr);
      } else {
        TensorEltwise(context,
                      type_, input0_ptr, input1_ptr, coeff_, input0->size(),
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
        TensorScalarEltwise(context,
                            type_, input0_ptr, input1_ptr[0], coeff_,
                            input0->size(), swapped, output_ptr);
      } else if (input0_shape == input1_shape) {
        TensorEltwise(context,
                      type_, input0_ptr, input1_ptr, coeff_, input0->size(),
                      swapped, output_ptr);
      } else if (need_general_broadcast) {
        TensorGeneralBroadcastEltwise(context,
                                      type_, input0_ptr, input1_ptr, coeff_,
                                      swapped, input0_shape, input1_shape,
                                      output_shape, output_ptr);
      } else {
        index_t common_size = input1->size();
        index_t diff_size = input0->size() / common_size;
        TensorBroadcastEltwise(context,
                               type_, input0_ptr, input1_ptr, coeff_,
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
  int has_data_format_;
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
            "scalar_input_index", 1))
#ifdef MACE_ENABLE_NEON
        , eltwise_(static_cast<ops::EltwiseType>(Operation::GetOptionalArg<int>(
            "type", static_cast<int>(ops::EltwiseType::NONE))))
#endif
  {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input0 = this->Input(0);
    MACE_CHECK(this->InputSize() == 2,
               "Quantized Elementwise don't support broadcast now.");
    const Tensor *input1 = this->Input(1);
    Tensor *output = this->Output(0);
    MACE_CHECK(type_ == SUM || type_ == SUB,
               "Quantized Elementwise only support SUM and SUB now.");
    MACE_CHECK(input0->size() == input1->size(),
               "input0 and input1 must have the same shape.");
    MACE_CHECK(output->scale() != 0);
    MACE_RETURN_IF_ERROR(output->Resize(input0->shape()));

#ifdef MACE_ENABLE_NEON
    eltwise_.Compute(context, input0, input1, output);
#else
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

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
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

        int32_t res;
        if (type_ == SUM) {
          res = multiplied_input0 + multiplied_input1;
        } else {
          res = multiplied_input0 - multiplied_input1;
        }

        const int32_t output_val =
            gemmlowp::RoundingDivideByPOT(
                gemmlowp::SaturatingRoundingDoublingHighMul(res,
                                                            output_multiplier),
                -output_shift) + output->zero_point();
        output_ptr[i] = Saturate<uint8_t>(output_val);
      }
    }, 0, output->size(), 1);
#endif  // NEON

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  Tensor scalar_tensor_;
#ifdef MACE_ENABLE_NEON
  arm::q8::Eltwise eltwise_;
#endif
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
    if (type == ops::EltwiseType::CLIP) {
      MACE_CHECK(coeff.size() == 2 && coeff[0] < coeff[1],
                 "Clip's min/max values are not correct.");
    }

    float scalar_input = Operation::GetOptionalArg<float>("scalar_input", 1.0);
    int32_t scalar_input_index = Operation::GetOptionalArg<int32_t>(
            "scalar_input_index", 1);
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::EltwiseKernel<T>>(
          type, coeff, scalar_input, scalar_input_index);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filters
    int input_size = operator_def_->input_size();
    Workspace *ws = context->workspace();
    for (int i = 0; i < input_size; ++i) {
      if (ws->HasTensor(operator_def_->input(i)) &&
          ws->GetTensor(operator_def_->input(i))->is_weight()) {
        if (ws->GetTensor(operator_def_->input(i))->dim_size() == 1) {
          MACE_CHECK(TransformFilter<T>(
              context,
              operator_def_.get(),
              i,
              OpenCLBufferType::ARGUMENT,
              mem_type) == MaceStatus::MACE_SUCCESS);
        } else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 4) {
          MACE_CHECK(TransformFilter<T>(
              context,
              operator_def_.get(),
              i,
              OpenCLBufferType::IN_OUT_CHANNEL,
              mem_type) == MaceStatus::MACE_SUCCESS);
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }
    }
  }
  MaceStatus Run(OpContext *context) override {
    MACE_CHECK(this->InputSize() < 3,
               "Element-Wise does not support 3 or higher inputs,"
               " you could change your model to multiple Element-Wise");
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

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
#include <utility>

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

inline void TensorScalar(const EltwiseType type,
                         const float *input0,
                         const float value,
                         const index_t size,
                         float *output) {
  switch (type) {
    case SUM:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] + value;
      }
      break;
    case SUB:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] - value;
      }
      break;
    case PROD:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * value;
      }
      break;
    case DIV:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] / value;
      }
      break;
    case MIN:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min<float>(input0[i], value);
      }
      break;
    case MAX:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max<float>(input0[i], value);
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
        output[i] = std::abs(input0[i]);
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - value, 2.f);
      }
      break;
    case POW:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i], value);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

inline void TensorBatchVector(const EltwiseType type,
                              const float *input0,
                              const float *input1,
                              const index_t batch,
                              const index_t channel,
                              const index_t hw,
                              const bool swapped,
                              float *output) {
  switch (type) {
    case SUM:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = input0[idx0] + input1[idx1];
          }
        }
      }
      break;
    case SUB:
      if (swapped) {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = b * channel + c;
              output[idx0] = input1[idx1] - input0[idx0];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = b * channel + c;
              output[idx0] = input0[idx0] - input1[idx1];
            }
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = input0[idx0] * input1[idx1];
          }
        }
      }
      break;
    case DIV:
      if (swapped) {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = b * channel + c;
              output[idx0] = input1[idx1] / input0[idx0];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = b * channel + c;
              output[idx0] = input0[idx0] / input1[idx1];
            }
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = std::min<float>(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = std::max<float>(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = std::pow(input0[idx0] - input1[idx1], 2.f);
          }
        }
      }
      break;
    case POW:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = b * channel + c;
            output[idx0] = std::pow(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}
inline void TensorVector(const EltwiseType type,
                         const float *input0,
                         const float *input1,
                         const index_t batch,
                         const index_t channel,
                         const index_t hw,
                         const bool swapped,
                         float *output) {
  switch (type) {
    case SUM:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = input0[idx0] + input1[idx1];
          }
        }
      }
      break;
    case SUB:
      if (swapped) {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = c;
              output[idx0] = input1[idx1] - input0[idx0];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = c;
              output[idx0] = input0[idx0] - input1[idx1];
            }
          }
        }
      }
      break;
    case PROD:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = input0[idx0] * input1[idx1];
          }
        }
      }
      break;
    case DIV:
      if (swapped) {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = c;
              output[idx0] = input1[idx1] / input0[idx0];
            }
          }
        }
      } else {
#pragma omp parallel for collapse(3)
        for (index_t b = 0; b < batch; ++b) {
          for (index_t c = 0; c < channel; ++c) {
            for (index_t i = 0; i < hw; ++i) {
              const index_t idx0 = (b * channel + c) * hw + i;
              const index_t idx1 = c;
              output[idx0] = input0[idx0] / input1[idx1];
            }
          }
        }
      }
      break;
    case MIN:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = std::min<float>(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    case MAX:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = std::max<float>(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = std::pow(input0[idx0] - input1[idx1], 2.f);
          }
        }
      }
      break;
    case POW:
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t i = 0; i < hw; ++i) {
            const index_t idx0 = (b * channel + c) * hw + i;
            const index_t idx1 = c;
            output[idx0] = std::pow(input0[idx0], input1[idx1]);
          }
        }
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}
inline void TensorEltwise(const EltwiseType type,
                          const float *input0,
                          const float *input1,
                          const index_t size,
                          float *output) {
  switch (type) {
    case SUM:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] + input1[i];
      }
      break;
    case SUB:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] - input1[i];
      }
      break;
    case PROD:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * input1[i];
      }
      break;
    case DIV:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] / input1[i];
      }
      break;
    case MIN:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::min<float>(input0[i], input1[i]);
      }
      break;
    case MAX:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::max<float>(input0[i], input1[i]);
      }
      break;
    case SQR_DIFF:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i] - input1[i], 2.f);
      }
      break;
    case POW:
#pragma omp parallel for
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i], input1[i]);
      }
      break;
    default:
      LOG(FATAL) << "Eltwise op not support type " << type;
  }
}


struct EltwiseFunctorBase {
  EltwiseFunctorBase(const EltwiseType type,
                     const std::vector<float> &coeff,
                     const float value)
      : type_(type), coeff_(coeff), value_(value) {}

  EltwiseType type_;
  std::vector<float> coeff_;
  float value_;
};

template <DeviceType D, typename T>
struct EltwiseFunctor;

template <>
struct EltwiseFunctor<DeviceType::CPU, float>: EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff,
                 const float value)
      : EltwiseFunctorBase(type, coeff, value) {}

  MaceStatus operator()(const Tensor *input0,
                  const Tensor *input1,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    bool swapped = false;
    if (input1 != nullptr) {
      MACE_CHECK(input0->dim_size() == input1->dim_size()
                     || input0->dim_size() == 1
                     || input1->dim_size() == 1)
        << "Inputs of Eltwise op must be same shape";
      if (input0->size() != input1->size()) {
        if (input0->size() < input1->size()) {
          std::swap(input0, input1);
          swapped = true;
        }
        if (input1->dim_size() == 1) {
          MACE_CHECK(input0->dim(1) == input1->dim(0))
            << "Element-Wise op only support channel dimension broadcast";
        } else {
          MACE_CHECK((input0->dim(0) == input1->dim(0) || input1->dim(0) == 1)
                         && input0->dim(1) == input1->dim(1)
                         && input1->dim(2) == 1
                         && input1->dim(3) == 1)
            << "Element-Wise op only support channel dimension broadcast";
        }
      }
    }
    MACE_FAILURE_RETURN(output->ResizeLike(input0));

    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard output_guard(output);

    const float *input0_ptr = input0->data<float>();
    float *output_ptr = output->mutable_data<float>();
    const index_t size = input0->size();
    if (input1 == nullptr) {
      TensorScalar(type_, input0_ptr, value_, size, output_ptr);
    } else {
      Tensor::MappingGuard input1_guard(input1);

      const float *input1_ptr = input1->data<float>();
      if (input1->size() != input0->size()) {
        const index_t batch = input0->dim(0);
        const index_t channel = input0->dim(1);
        const index_t hw = input0->dim(2) * input0->dim(3);
        if (input1->dim(0) == 1 || input1->dim_size() == 1)
          TensorVector(type_, input0_ptr, input1_ptr,
                       batch, channel, hw, swapped, output_ptr);
        else
          TensorBatchVector(type_, input0_ptr, input1_ptr,
                            batch, channel, hw, swapped, output_ptr);
      } else {
        if (!coeff_.empty() && type_ == SUM) {
#pragma omp parallel for
          for (index_t i = 0; i < size; ++i) {
            output_ptr[i] = coeff_[0] * input0_ptr[i] +
                coeff_[1] * input1_ptr[i];
          }
        } else {
          TensorEltwise(type_, input0_ptr, input1_ptr, size, output_ptr);
        }
      }
    }

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct EltwiseFunctor<DeviceType::GPU, T> : EltwiseFunctorBase {
  EltwiseFunctor(const EltwiseType type,
                 const std::vector<float> &coeff,
                 const float value)
      : EltwiseFunctorBase(type, coeff, value) {}

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

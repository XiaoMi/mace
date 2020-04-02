// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MICRO_OPS_ELTWISE_H_
#define MICRO_OPS_ELTWISE_H_

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

namespace eltwise {  // for redefine
enum Type {
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
  EQUAL = 10,
  FLOOR_DIV = 11,
  CLIP = 12,
  SIGN = 13,
  NONE = 14,
};

bool ShapeIsEqual(const int32_t *dims0,
                  const int32_t *dims1, uint32_t dim_size);
int32_t GetIndex(const int32_t *shape, const int32_t *index, int32_t dim_size);
void IncreaseIndex(const int32_t *shape, int32_t **index, int32_t dim_size);
template<typename T>
int32_t Sign(T val) {
  return (T(0) < val) - (val < T(0));
}

}  // namespace eltwise

template<typename T>
class EltwiseOp : public framework::Operator {
 public:
  MaceStatus OnInit() {
    input0_ = GetInputData<T>(INPUT0);
    input0_dims_ = GetInputShapeDims(INPUT0);
    input0_dim_size_ = GetInputShapeDimSize(INPUT0);

    if (GetInputSize() >= 2) {
      input1_ = GetInputData<T>(INPUT1);
      input1_dims_ = GetInputShapeDims(INPUT1);
      input1_dim_size_ = GetInputShapeDimSize(INPUT1);
    } else {
      input1_ = NULL;
      input1_dims_ = NULL;
      input1_dim_size_ = 0;
    }

    output_ = GetOutputData<T>(OUTPUT);

    type_ = static_cast<eltwise::Type>(GetArgByName(
        "type", static_cast<int32_t>(NONE)));
    coeff_ = GetRepeatArgByName<float>("coeff", &coeff_size_);
    scalar_input_ = GetArgByName("scalar_input", 1.0f);
    scalar_input_index_ = GetArgByName("scalar_input_index",
                                       static_cast<int32_t>(1));
    DataFormat data_format = static_cast<DataFormat>(
        GetArgByName("data_format", static_cast<int32_t>(NHWC)));
    nchw_ = (data_format == NCHW);

    return MACE_SUCCESS;
  }

  MaceStatus Run() {
    MACE_ASSERT1(GetInputSize() < 3,
                 "Element-Wise does not support 3 or higher inputs,"
                 " you could change your model to multiple Element-Wise");

    if (input1_ == NULL) {
      input1_ = &scalar_input_;
      input1_dim_size_ = 1;
      input1_dims_ = static_cast<const int32_t *>(
          reinterpret_cast<int32_t *>(&input1_dim_size_));  // a trick
    }

    if (type_ == eltwise::CLIP) {
      MACE_ASSERT1(coeff_size_ == 2 && coeff_[0] < coeff_[1],
                   "Clip's min/max values are not correct.");
    }

    if (type_ == eltwise::EQUAL) {  // IsLogicalType
      // as we do not have bool-type tensor, we use int type
      return DoEltwise<int32_t>();
    } else {
      return DoEltwise<T>();
    }
  }

 private:
  template<typename DstType>
  MaceStatus DoEltwise() {
    int32_t input0_size = base::GetShapeSize(input0_dim_size_, input0_dims_);
    int32_t input1_size = input1_dim_size_ == 0 ?
                          0 : base::GetShapeSize(input1_dim_size_,
                                                 input1_dims_);
    bool swapped = false;
    if (input0_dim_size_ < input1_dim_size_
        || (input0_dim_size_ == input1_dim_size_
            && input0_size < input1_size)) {
      base::swap(&input0_, &input1_);
      base::swap(&input0_dims_, &input1_dims_);
      base::swap(&input0_dim_size_, &input1_dim_size_);
      base::swap(&input0_size, &input1_size);
      swapped = true;
    }
    if (scalar_input_index_ == 0) {
      swapped = !swapped;
    }

    // check if we can broadcast tensor
    uint32_t rank_diff =
        static_cast<uint32_t>(input0_dim_size_ - input1_dim_size_);
    if (nchw_) {
      MACE_ASSERT1((input0_dim_size_ == 4) &&
          ((input1_dim_size_ == 0) ||
              (input1_dim_size_ == 4 && input1_dims_[1] == input0_dims_[1] &&
                  (input1_dims_[0] == input0_dims_[0] ||
                      input1_dims_[0] == 1)) ||
              (input1_dim_size_ == 1 && input1_dims_[0] == input0_dims_[1])),
                   "only support broadcast channel dimension");
    } else {
      for (uint32_t i = 0; i < input1_dim_size_; ++i) {
        MACE_ASSERT1(input0_dims_[rank_diff + i] == 1 || input1_dims_[i] == 1 ||
            input0_dims_[rank_diff + i] == input1_dims_[i],
                     "Element-Wise op only support tail dimensions broadcast");
      }
    }

    if (nchw_ && input1_dim_size_ > 0) {
      MACE_RETURN_IF_ERROR(
          ResizeOutputShape(OUTPUT, input0_dim_size_, input0_dims_));
      DstType *output_ptr = reinterpret_cast<DstType *>(output_);
      if (input1_size < input0_size) {
        TensorEltwisePerChannel(type_,
                                input0_,
                                input1_,
                                input0_dims_[0],
                                input1_dim_size_ == 1 ? 1 : input1_dims_[0],
                                input0_dims_[1],
                                input0_dims_[2] * input0_dims_[3],
                                swapped,
                                output_ptr);
      } else {
        TensorEltwise(type_, input0_, input1_, input0_size,
                      swapped, output_ptr);
      }
    } else {
      ScratchBuffer scratch_buffer(engine_config_);
      int32_t *input1_shape =
          scratch_buffer.GetBuffer<int32_t>(input0_dim_size_);
      if (rank_diff > 0) {
        base::memset(input1_shape, static_cast<int32_t>(1), rank_diff);
      }
      if (input1_dim_size_ > 0) {
        base::memcpy(input1_shape + rank_diff, input1_dims_,
                     input1_dim_size_ * sizeof(int32_t));
      }

      int32_t *output_shape =
          scratch_buffer.GetBuffer<int32_t>(input0_dim_size_);
      for (uint32_t i = 0; i < input0_dim_size_; ++i) {
        output_shape[i] = base::max(input0_dims_[i], input1_shape[i]);
      }
      MACE_RETURN_IF_ERROR(
          ResizeOutputShape(OUTPUT, input0_dim_size_, output_shape));

      DstType *output_ptr = reinterpret_cast<DstType *>(output_);
      bool need_general_broadcast = false;
      for (uint32_t i = 0; i < input1_dim_size_; ++i) {
        if ((input0_dims_[rank_diff + i] == 1 && input1_dims_[i] > 1) ||
            (input0_dims_[rank_diff + i] > 1 && input1_dims_[i] == 1)) {
          need_general_broadcast = true;
          break;
        }
      }

      if (input1_size == 1) {
        TensorScalarEltwise(type_, input0_, input1_[0],
                            input0_size, swapped, output_ptr);
      } else if (eltwise::ShapeIsEqual(input0_dims_,
                                       input1_shape,
                                       input0_dim_size_)) {
        TensorEltwise(type_, input0_, input1_, input0_size,
                      swapped, output_ptr);
      } else if (need_general_broadcast) {
        int32_t *out_index =
            scratch_buffer.GetBuffer<int32_t>(input0_dim_size_);
        TensorGeneralBroadcastEltwise(type_, input0_, input1_, input0_dim_size_,
                                      swapped, input0_dims_, input1_shape,
                                      output_shape, out_index, output_ptr);
      } else {
        int32_t common_size = input1_size;
        int32_t diff_size = input0_size / common_size;
        TensorBroadcastEltwise(type_, input0_, input1_,
                               diff_size, common_size, swapped, output_ptr);
      }
    }

    return MACE_SUCCESS;
  }

  template<typename DstType>
  inline void TensorGeneralBroadcastEltwise(
      const eltwise::Type type,
      const T *input0,
      const T *input1,
      const uint32_t dim_size,
      const bool swapped,
      const int32_t *input0_shape,
      const int32_t *input1_shape,
      const int32_t *output_shape,
      int32_t *out_index,
      DstType *output) {
    const int32_t output_size = base::GetShapeSize(dim_size, output_shape);
    base::memset(out_index, static_cast<int32_t>(0), dim_size);
    switch (type) {
      case eltwise::SUM:
        if (coeff_size_ == 0) {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = input0[idx0] + input1[idx1];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        } else {
          float coeff_copy[2] = {coeff_[0], coeff_[1]};
          if (swapped) {
            base::swap(coeff_copy, coeff_copy + 1);
          }
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] =
                input0[idx0] * coeff_copy[0] + input1[idx1] * coeff_copy[1];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        }
        break;
      case eltwise::SUB:
        if (!swapped) {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = input0[idx0] - input1[idx1];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        } else {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = input1[idx1] - input0[idx0];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        }
        break;
      case eltwise::PROD:
        for (int32_t i = 0; i < output_size; ++i) {
          const int32_t idx0 =
              eltwise::GetIndex(input0_shape, out_index, dim_size);
          const int32_t idx1 =
              eltwise::GetIndex(input1_shape, out_index, dim_size);
          output[i] = input0[idx0] * input1[idx1];
          eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
        }
        break;
      case eltwise::DIV:
        if (!swapped) {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = input0[idx0] / input1[idx1];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        } else {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = input1[idx1] / input0[idx0];
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        }
        break;
      case eltwise::FLOOR_DIV:
        if (!swapped) {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = base::floor(input0[idx0] / input1[idx1]);
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        } else {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = base::floor(input1[idx1] / input0[idx0]);
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        }
        break;
      case eltwise::MIN:
        for (int32_t i = 0; i < output_size; ++i) {
          const int32_t idx0 =
              eltwise::GetIndex(input0_shape, out_index, dim_size);
          const int32_t idx1 =
              eltwise::GetIndex(input1_shape, out_index, dim_size);
          output[i] = base::min(input1[idx1], input0[idx0]);
          eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
        }
        break;
      case eltwise::MAX:
        for (int32_t i = 0; i < output_size; ++i) {
          const int32_t idx0 =
              eltwise::GetIndex(input0_shape, out_index, dim_size);
          const int32_t idx1 =
              eltwise::GetIndex(input1_shape, out_index, dim_size);
          output[i] = base::max(input1[idx1], input0[idx0]);
          eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
        }
        break;
      case eltwise::SQR_DIFF:
        for (int32_t i = 0; i < output_size; ++i) {
          const int32_t idx0 =
              eltwise::GetIndex(input0_shape, out_index, dim_size);
          const int32_t idx1 =
              eltwise::GetIndex(input1_shape, out_index, dim_size);
          output[i] = base::pow(input1[idx1] - input0[idx0], 2.f);
          eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
        }
        break;
      case eltwise::POW:
        if (!swapped) {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = base::pow(input0[idx0], input1[idx1]);
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        } else {
          for (int32_t i = 0; i < output_size; ++i) {
            const int32_t idx0 =
                eltwise::GetIndex(input0_shape, out_index, dim_size);
            const int32_t idx1 =
                eltwise::GetIndex(input1_shape, out_index, dim_size);
            output[i] = base::pow(input1[idx1], input0[idx0]);
            eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
          }
        }
        break;
      case eltwise::EQUAL:
        for (int32_t i = 0; i < output_size; ++i) {
          const int32_t idx0 =
              eltwise::GetIndex(input0_shape, out_index, dim_size);
          const int32_t idx1 =
              eltwise::GetIndex(input1_shape, out_index, dim_size);
          output[i] = input1[idx1] == input0[idx0];
          eltwise::IncreaseIndex(output_shape, &out_index, dim_size);
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type "
                         << static_cast<int32_t>(type);
    }
  }

  template<typename DstType>
  inline void TensorBroadcastEltwise(const eltwise::Type type,
                                     const T *input0,
                                     const T *input1,
                                     const int32_t diff_size,
                                     const int32_t common_size,
                                     const bool swapped,
                                     DstType *output) {
    switch (type) {
      case eltwise::SUM:
        if (coeff_size_ == 0) {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input0[i + d * common_size] + input1[i];
            }
          }
        } else {
          float coeff_copy[2] = {coeff_[0], coeff_[1]};
          if (swapped) {
            base::swap(coeff_copy, coeff_copy + 1);
          }
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input0[i + d * common_size] * coeff_copy[0] +
                      input1[i] * coeff_copy[1];
            }
          }
        }
        break;
      case eltwise::SUB:
        if (!swapped) {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input0[i + d * common_size] - input1[i];
            }
          }
        } else {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input1[i] - input0[i + d * common_size];
            }
          }
        }
        break;
      case eltwise::PROD:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] * input1[i];
          }
        }
        break;
      case eltwise::DIV:
        if (!swapped) {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input0[i + d * common_size] / input1[i];
            }
          }
        } else {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  input1[i] / input0[i + d * common_size];
            }
          }
        }
        break;
      case eltwise::FLOOR_DIV:
        if (!swapped) {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  base::floor(input0[i + d * common_size] / input1[i]);
            }
          }
        } else {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  base::floor(input1[i] / input0[i + d * common_size]);
            }
          }
        }
        break;
      case eltwise::MIN:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                base::min(input0[i + d * common_size], input1[i]);
          }
        }
        break;
      case eltwise::MAX:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                base::max(input0[i + d * common_size], input1[i]);
          }
        }
        break;
      case eltwise::SQR_DIFF:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                base::pow(input0[i + d * common_size] - input1[i], 2.f);
          }
        }
        break;
      case eltwise::POW:
        if (!swapped) {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  base::pow(input0[i + d * common_size], input1[i]);
            }
          }
        } else {
          for (int32_t d = 0; d < diff_size; ++d) {
            for (int32_t i = 0; i < common_size; ++i) {
              output[i + d * common_size] =
                  base::pow(input1[i], input0[i + d * common_size]);
            }
          }
        }
        break;
      case eltwise::NEG:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] = -input0[i + d * common_size];
          }
        }
        break;
      case eltwise::ABS:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                base::fabs(input0[i + d * common_size]);
          }
        }
        break;
      case eltwise::EQUAL:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                input0[i + d * common_size] == input1[i];
          }
        }
        break;
      case eltwise::CLIP:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                base::max<float>(coeff_[0],
                                 base::min<float>(coeff_[1],
                                                  input0[i + d * common_size]));
          }
        }
        break;
      case eltwise::SIGN:
        for (int32_t d = 0; d < diff_size; ++d) {
          for (int32_t i = 0; i < common_size; ++i) {
            output[i + d * common_size] =
                eltwise::Sign(input0[i + d * common_size]);
          }
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type "
                         << static_cast<int32_t>(type);
    }
  }

// Multiplication is costly, so we specialize the following case.
  template<typename DstType>
  inline void TensorEltwise(const eltwise::Type type,
                            const T *input0,
                            const T *input1,
                            const int32_t size,
                            const bool swapped,
                            DstType *output) {
    switch (type) {
      case eltwise::SUM:
        if (coeff_size_ == 0) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] + input1[i];
          }
        } else {
          float coeff_copy[2] = {coeff_[0], coeff_[1]};
          if (swapped) {
            base::swap(coeff_copy, coeff_copy + 1);
          }
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] * coeff_copy[0] + input1[i] * coeff_copy[1];
          }
        }
        break;
      case eltwise::SUB:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] - input1[i];
          }
        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input1[i] - input0[i];
          }
        }
        break;
      case eltwise::PROD:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = input0[i] * input1[i];
        }
        break;
      case eltwise::DIV:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] / input1[i];
          }

        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input1[i] / input0[i];
          }
        }
        break;
      case eltwise::FLOOR_DIV:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::floor(input0[i] / input1[i]);
          }
        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::floor(input1[i] / input0[i]);
          }
        }
        break;
      case eltwise::MIN:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::min(input0[i], input1[i]);
        }
        break;
      case eltwise::MAX:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::max(input0[i], input1[i]);
        }
        break;
      case eltwise::SQR_DIFF:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::pow(input0[i] - input1[i], 2.f);
        }
        break;
      case eltwise::POW:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::pow(input0[i], input1[i]);
          }
        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::pow(input1[i], input0[i]);
          }
        }
        break;
      case eltwise::NEG:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = -input0[i];
        }
        break;
      case eltwise::ABS:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::fabs(input0[i]);
        }
        break;
      case eltwise::EQUAL:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = (input0[i] == input1[i]);
        }
        break;
      case eltwise::CLIP:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::max<float>(
              coeff_[0], base::min<float>(coeff_[1], input0[i]));
        }
        break;
      case eltwise::SIGN:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = eltwise::Sign(input0[i]);
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type "
                         << static_cast<int32_t>(type);
    }
  }

// Multiplication is costly, so we specialize the following case.
  template<typename DstType>
  inline void TensorScalarEltwise(const eltwise::Type type,
                                  const T *input0,
                                  const T input1,
                                  const int32_t size,
                                  const bool swapped,
                                  DstType *output) {
    switch (type) {
      case eltwise::SUM:
        if (coeff_size_ == 0) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] + input1;
          }

        } else {
          float coeff_copy[2] = {coeff_[0], coeff_[1]};
          if (swapped) {
            base::swap(coeff_copy, coeff_copy + 1);
          }
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] * coeff_copy[0] + input1 * coeff_copy[1];
          }
        }
        break;
      case eltwise::SUB:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] - input1;
          }

        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input1 - input0[i];
          }
        }
        break;
      case eltwise::PROD:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = input0[i] * input1;
        }
        break;
      case eltwise::DIV:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input0[i] / input1;
          }

        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = input1 / input0[i];
          }
        }
        break;
      case eltwise::FLOOR_DIV:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::floor(input0[i] / input1);
          }
        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::floor(input1 / input0[i]);
          }
        }
        break;
      case eltwise::MIN:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::min(input0[i], input1);
        }

        break;
      case eltwise::MAX:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::max(input0[i], input1);
        }

        break;
      case eltwise::SQR_DIFF:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::pow(input0[i] - input1, 2.f);
        }

        break;
      case eltwise::POW:
        if (!swapped) {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::pow(input0[i], input1);
          }
        } else {
          for (int32_t i = 0; i < size; ++i) {
            output[i] = base::pow(input1, input0[i]);
          }
        }
        break;
      case eltwise::NEG:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = -input0[i];
        }
        break;
      case eltwise::ABS:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::fabs(input0[i]);
        }
        break;
      case eltwise::EQUAL:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = (input0[i] == input1);
        }
        break;
      case eltwise::CLIP:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = base::max<float>(coeff_[0],
                                       base::min<float>(coeff_[1], input0[i]));
        }
        break;
      case eltwise::SIGN:
        for (int32_t i = 0; i < size; ++i) {
          output[i] = eltwise::Sign(input0[i]);
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type "
                         << static_cast<int32_t>(type);
    }
  }

  template<typename DstType>
  inline void TensorEltwisePerChannel(const eltwise::Type type,
                                      const T *input0,
                                      const T *input1,
                                      const int32_t batch0,
                                      const int32_t batch1,
                                      const int32_t channel,
                                      const int32_t image_size,
                                      const bool swapped,
                                      DstType *output) {
    switch (type) {
      case eltwise::SUM:
        if (coeff_size_ == 0) {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in0_ptr[i] + in1_ptr[c];
              }
            }
          }
        } else {
          float coeff_copy[2] = {coeff_[0], coeff_[1]};
          if (swapped) {
            base::swap(coeff_copy, coeff_copy + 1);  // NOLINT
          }
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] =
                    in0_ptr[i] * coeff_copy[0] + in1_ptr[c] * coeff_copy[1];
              }
            }
          }
        }
        break;
      case eltwise::SUB:
        if (!swapped) {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in0_ptr[i] - in1_ptr[c];
              }
            }
          }
        } else {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in1_ptr[c] - in0_ptr[i];
              }
            }
          }
        }
        break;
      case eltwise::PROD:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] * in1_ptr[c];
            }
          }
        }
        break;
      case eltwise::DIV:
        if (!swapped) {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in0_ptr[i] / in1_ptr[c];
              }
            }
          }
        } else {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = in1_ptr[c] / in0_ptr[i];
              }
            }
          }
        }
        break;
      case eltwise::FLOOR_DIV:
        if (!swapped) {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = base::floor(in0_ptr[i] / in1_ptr[c]);
              }
            }
          }
        } else {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = base::floor(in1_ptr[c] / in0_ptr[i]);
              }
            }
          }
        }
        break;
      case eltwise::MIN:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = base::min(in0_ptr[i], in1_ptr[c]);
            }
          }
        }
        break;
      case eltwise::MAX:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = base::max(in0_ptr[i], in1_ptr[c]);  // NOLINT
            }
          }
        }
        break;
      case eltwise::SQR_DIFF:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = base::pow(in0_ptr[i] - in1_ptr[c], 2.f);
            }
          }
        }
        break;
      case eltwise::POW:
        if (!swapped) {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = base::pow(in0_ptr[i], in1_ptr[c]);
              }
            }
          }
        } else {
          for (int32_t b = 0; b < batch0; ++b) {
            for (int32_t c = 0; c < channel; ++c) {
              const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
              const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
              DstType *out_ptr = output + ((b * channel) + c) * image_size;
              for (int32_t i = 0; i < image_size; ++i) {
                out_ptr[i] = base::pow(in1_ptr[c], in0_ptr[i]);
              }
            }
          }
        }
        break;
      case eltwise::NEG:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = -input0[i];
            }
          }
        }
        break;
      case eltwise::ABS:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            for (int32_t i = 0; i < image_size; ++i) {
              output[i] = base::fabs(input0[i]);
            }
          }
        }
        break;
      case eltwise::EQUAL:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
            const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
            DstType *out_ptr = output + ((b * channel) + c) * image_size;
            for (int32_t i = 0; i < image_size; ++i) {
              out_ptr[i] = in0_ptr[i] == in1_ptr[c];
            }
          }
        }
        break;
      case eltwise::SIGN:
        for (int32_t b = 0; b < batch0; ++b) {
          for (int32_t c = 0; c < channel; ++c) {
            for (int32_t i = 0; i < image_size; ++i) {
              output[i] = eltwise::Sign(input0[i]);
            }
          }
        }
        break;
      default:LOG(FATAL) << "Eltwise op not support type "
                         << static_cast<int32_t>(type);
    }
  }

 private:
  const T *input0_;
  const int32_t *input0_dims_;
  uint32_t input0_dim_size_;

  const T *input1_;
  const int32_t *input1_dims_;
  uint32_t input1_dim_size_;

  T *output_;

  eltwise::Type type_;
  const float *coeff_;
  uint32_t coeff_size_;
  T scalar_input_;
  int32_t scalar_input_index_;
  bool nchw_;

  MACE_OP_INPUT_TAGS(INPUT0, INPUT1);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_ELTWISE_H_

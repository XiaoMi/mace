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

#ifndef MICRO_OPS_REDUCE_H_
#define MICRO_OPS_REDUCE_H_

#include "micro/base/logging.h"
#include "micro/base/types.h"
#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

class ReduceOpBase : public framework::Operator {
 public:
  MaceStatus OnInit();

 public:
  enum ReduceType {
    MEAN = 0,
    MIN = 1,
    MAX = 2,
    PROD = 3,
    SUM = 4,
  };

 protected:
  void Validate();

 protected:
  ReduceType reduce_type_;
  const int32_t *axis_;
  uint32_t axis_size_;
  bool keep_dims_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

template<typename T>
class ReduceOp : public ReduceOpBase {
 public:
  MaceStatus OnInit() {
    input_ = GetInputData<T>(INPUT);
    input_dims_ = GetInputShapeDims(INPUT);
    input_dim_size_ = GetInputShapeDimSize(INPUT);

    output_ = GetOutputData<T>(OUTPUT);
    return ReduceOpBase::OnInit();
  }

  MaceStatus Run() {
    Validate();
    ScratchBuffer scratch_buffer(engine_config_);
    bool *bitmap = scratch_buffer.GetBuffer<bool>(input_dim_size_);
    int32_t *data_dims = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    uint32_t data_dim_size = 0;
    int32_t *output_dims = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    uint32_t output_dim_size = 0;
    Simplify(output_dims, &output_dim_size, bitmap,
             input_dim_size_, data_dims, &data_dim_size);
    MACE_RETURN_IF_ERROR(
        ResizeOutputShape(OUTPUT, output_dim_size, output_dims));
    const int32_t output_size =
        base::GetShapeSize(output_dim_size, output_dims);
    Compute(data_dims, data_dim_size, static_cast<uint32_t >(output_size));

    return MACE_SUCCESS;
  }

 private:
  void Simplify(int32_t *output_dims, uint32_t *output_dim_size,
                bool *bitmap, int32_t bitmap_size,
                int32_t *data_dims, uint32_t *data_dim_size) {
    base::memset(bitmap, false, bitmap_size);
    if (axis_size_ == 0) {
      for (uint32_t i = 0; i < input_dim_size_; ++i) {
        bitmap[i] = true;
      }
    } else {
      for (uint32_t i = 0; i < axis_size_; ++i) {
        int32_t index = axis_[i] >= 0 ? axis_[i] : axis_[i] + input_dim_size_;
        DataFormat data_format = static_cast<DataFormat>(GetArgByName(
            "data_format", static_cast<int32_t >(NHWC)));
        if (data_format == NCHW &&
            DataTypeToEnum<T>::value != DT_UINT8 && input_dim_size_ == 4) {
          if (index == 1 || index == 2) {
            index = index + 1;
          } else if (index == 3) {
            index = 1;
          }
        }
        bitmap[index] = true;
      }
    }
    uint32_t out_dim_idx = 0;
    for (uint32_t i = 0; i < input_dim_size_; ++i) {
      if (!bitmap[i]) {
        output_dims[out_dim_idx++] = input_dims_[i];
      } else if (keep_dims_) {
        output_dims[out_dim_idx++] = 1;
      }
    }
    *output_dim_size = out_dim_idx;

    int32_t data_dims_idx = 0;
    uint32_t dim_index = 0;
    for (; dim_index < input_dim_size_; ++dim_index) {
      if (input_dims_[dim_index] != 1) break;
    }
    if (dim_index >= input_dim_size_) {
      reduce_first_axis_ = true;
    } else {
      reduce_first_axis_ = bitmap[dim_index];
      data_dims[data_dims_idx++] = input_dims_[dim_index];
      ++dim_index;
      for (; dim_index < input_dim_size_; ++dim_index) {
        const int32_t n = input_dims_[dim_index];
        if (n == 1) {
          bitmap[dim_index] = bitmap[dim_index - 1];
        }
        if (bitmap[dim_index - 1] != bitmap[dim_index]) {
          data_dims[data_dims_idx++] = n;
        } else {
          data_dims[data_dims_idx - 1] *= n;
        }
      }
    }
    *data_dim_size = data_dims_idx;
  }

  void Reduce1Dims(ReduceType type, int32_t *data_reshape) {
    if (reduce_first_axis_) {
      if (type == MEAN) {
        T tmp = 0;
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          tmp = tmp + input_[i];
        }
        output_[0] = tmp / data_reshape[0];
      } else if (type == MIN) {
        T tmp = input_[0];
        for (int32_t i = 1; i < data_reshape[0]; ++i) {
          tmp = base::min<T>(tmp, input_[i]);
        }
        output_[0] = tmp;
      } else if (type == MAX) {
        T tmp = input_[0];
        for (int32_t i = 1; i < data_reshape[0]; ++i) {
          tmp = base::max<T>(tmp, input_[i]);
        }
        output_[0] = tmp;
      } else if (type == PROD) {
        T tmp = input_[0];
        for (int32_t i = 1; i < data_reshape[0]; ++i) {
          tmp = tmp * input_[i];
        }
        output_[0] = tmp;
      } else if (type == SUM) {
        T tmp = 0;
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          tmp = tmp + input_[i];
        }
        output_[0] = tmp;
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      base::memcpy(output_, input_, data_reshape[0] * sizeof(T));
    }
  }

  void Reduce2Dims(ReduceType type, int32_t *data_reshape) {
    if (reduce_first_axis_) {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = 0;
          for (int32_t j = 0; j < data_reshape[0]; ++j) {
            tmp += input_[j * data_reshape[1] + i];
          }
          output_[i] = tmp / data_reshape[0];
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = input_[i];
          for (int32_t j = 1; j < data_reshape[0]; ++j) {
            tmp = base::min(tmp, input_[j * data_reshape[1] + i]);
          }
          output_[i] = tmp;
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = input_[i];
          for (int32_t j = 1; j < data_reshape[0]; ++j) {
            tmp = base::max(tmp, input_[j * data_reshape[1] + i]);
          }
          output_[i] = tmp;
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = input_[i];
          for (int32_t j = 1; j < data_reshape[0]; ++j) {
            tmp = tmp * input_[j * data_reshape[1] + i];
          }
          output_[i] = tmp;
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = 0;
          for (int32_t j = 0; j < data_reshape[0]; ++j) {
            tmp += input_[j * data_reshape[1] + i];
          }
          output_[i] = tmp;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          T tmp = 0;
          for (int32_t j = 0; j < data_reshape[1]; ++j) {
            tmp += input_[i * data_reshape[1] + j];
          }
          output_[i] = tmp / data_reshape[1];
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          T tmp = input_[i * data_reshape[1]];
          for (int32_t j = 1; j < data_reshape[1]; ++j) {
            tmp = base::min(tmp, input_[i * data_reshape[1] + j]);
          }
          output_[i] = tmp;
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          T tmp = input_[i * data_reshape[1]];
          for (int32_t j = 1; j < data_reshape[1]; ++j) {
            tmp = base::max(tmp, input_[i * data_reshape[1] + j]);
          }
          output_[i] = tmp;
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          T tmp = input_[i * data_reshape[1]];
          for (int32_t j = 1; j < data_reshape[1]; ++j) {
            tmp = tmp * input_[i * data_reshape[1] + j];
          }
          output_[i] = tmp;
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          T tmp = 0;
          for (int32_t j = 0; j < data_reshape[1]; ++j) {
            tmp += input_[i * data_reshape[1] + j];
          }
          output_[i] = tmp;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }
  }

  void Reduce3Dims(ReduceType type, int32_t *data_reshape) {
    if (reduce_first_axis_) {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[0]; ++k) {
              output_[i] +=
                  input_[(k * data_reshape[1] + i) * data_reshape[2]
                      + j];
            }
          }
          output_[i] /= (data_reshape[0] * data_reshape[2]);
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = input_[i * data_reshape[2]];
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[0]; ++k) {
              tmp = base::min(
                  tmp, input_[(k * data_reshape[1] + i) * data_reshape[2] + j]);
            }
          }
          output_[i] = tmp;
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = input_[i * data_reshape[2]];
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[0]; ++k) {
              tmp = base::max(
                  tmp, input_[(k * data_reshape[1] + i) * data_reshape[2] + j]);
            }
          }
          output_[i] = tmp;
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          T tmp = 1;
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[0]; ++k) {
              tmp *= input_[(k * data_reshape[1] + i) * data_reshape[2] + j];
            }
          }
          output_[i] = tmp;
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[0]; ++k) {
              output_[i] +=
                  input_[(k * data_reshape[1] + i) * data_reshape[2] + j];
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              output_[i * data_reshape[2] + j] +=
                  input_[(i * data_reshape[1] + k) * data_reshape[2] + j];
            }
            output_[i * data_reshape[2] + j] /= data_reshape[1];
          }
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = input_[i * data_reshape[1] * data_reshape[2] + j];
            for (int32_t k = 1; k < data_reshape[1]; ++k) {
              tmp = base::min(
                  tmp, input_[(i * data_reshape[1] + k) * data_reshape[2] + j]);
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = input_[i * data_reshape[1] * data_reshape[2] + j];
            for (int32_t k = 1; k < data_reshape[1]; ++k) {
              tmp = base::max(
                  tmp, input_[(i * data_reshape[1] + k) * data_reshape[2] + j]);
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = input_[i * data_reshape[1] * data_reshape[2] + j];
            for (int32_t k = 1; k < data_reshape[1]; ++k) {
              tmp *= input_[(i * data_reshape[1] + k) * data_reshape[2] + j];
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              output_[i * data_reshape[2] + j] +=
                  input_[(i * data_reshape[1] + k) * data_reshape[2] + j];
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }
  }

  void Reduce4Dims(ReduceType type, int32_t *data_reshape) {
    if (reduce_first_axis_) {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[3]; ++j) {
            for (int32_t k = 0; k < data_reshape[2]; ++k) {
              for (int32_t t = 0; t < data_reshape[0]; ++t) {
                output_[i * data_reshape[3] + j] +=
                    input_[((t * data_reshape[1] + i) *
                        data_reshape[2] + k) * data_reshape[3] + j];
              }
            }
            output_[i * data_reshape[3] + j] /=
                (data_reshape[0] * data_reshape[2]);
          }
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[3]; ++j) {
            T tmp = input_[i * data_reshape[2] * data_reshape[3] + j];
            for (int32_t k = 0; k < data_reshape[2]; ++k) {
              for (int32_t t = 0; t < data_reshape[0]; ++t) {
                tmp = base::min(tmp,
                                input_[((t * data_reshape[1] + i) *
                                    data_reshape[2] + k) * data_reshape[3]
                                    + j]);
              }
            }
            output_[i * data_reshape[3] + j] = tmp;
          }
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[3]; ++j) {
            T tmp = input_[i * data_reshape[2] * data_reshape[3] + j];
            for (int32_t k = 0; k < data_reshape[2]; ++k) {
              for (int32_t t = 0; t < data_reshape[0]; ++t) {
                tmp = base::max(tmp,  // NOLINT
                                input_[((t * data_reshape[1] + i) *
                                    data_reshape[2] + k) * data_reshape[3]
                                    + j]);
              }
            }
            output_[i * data_reshape[3] + j] = tmp;
          }
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[3]; ++j) {
            T tmp = 1;
            for (int32_t k = 0; k < data_reshape[2]; ++k) {
              for (int32_t t = 0; t < data_reshape[0]; ++t) {
                tmp = tmp * input_[((t * data_reshape[1] + i) *
                    data_reshape[2] + k) * data_reshape[3] + j];
              }
            }
            output_[i * data_reshape[3] + j] = tmp;
          }
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[1]; ++i) {
          for (int32_t j = 0; j < data_reshape[3]; ++j) {
            for (int32_t k = 0; k < data_reshape[2]; ++k) {
              for (int32_t t = 0; t < data_reshape[0]; ++t) {
                output_[i * data_reshape[3] + j] +=
                    input_[((t * data_reshape[1] + i) *
                        data_reshape[2] + k) * data_reshape[3] + j];
              }
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      if (type == MEAN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              for (int32_t t = 0; t < data_reshape[3]; ++t) {
                output_[i * data_reshape[2] + j] +=
                    input_[((i * data_reshape[1] + k) *
                        data_reshape[2] + j) * data_reshape[3] + t];
              }
            }
            output_[i * data_reshape[2] + j] /=
                (data_reshape[1] * data_reshape[3]);
          }
        }
      } else if (type == MIN) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = input_[(i * data_reshape[1] *
                data_reshape[2] + j) * data_reshape[3]];
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              for (int32_t t = 0; t < data_reshape[3]; ++t) {
                tmp = base::min(
                    tmp, input_[((i * data_reshape[1] + k) *
                        data_reshape[2] + j) * data_reshape[3] + t]);
              }
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == MAX) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = input_[(i * data_reshape[1] *
                data_reshape[2] + j) * data_reshape[3]];
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              for (int32_t t = 0; t < data_reshape[3]; ++t) {
                tmp = base::max(
                    tmp, input_[((i * data_reshape[1] + k) *
                        data_reshape[2] + j) * data_reshape[3] + t]);
              }
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == PROD) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            T tmp = 1;
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              for (int32_t t = 0; t < data_reshape[3]; ++t) {
                tmp = tmp * input_[((i * data_reshape[1] + k) *
                    data_reshape[2] + j) * data_reshape[3] + t];
              }
            }
            output_[i * data_reshape[2] + j] = tmp;
          }
        }
      } else if (type == SUM) {
        for (int32_t i = 0; i < data_reshape[0]; ++i) {
          for (int32_t j = 0; j < data_reshape[2]; ++j) {
            for (int32_t k = 0; k < data_reshape[1]; ++k) {
              for (int32_t t = 0; t < data_reshape[3]; ++t) {
                output_[i * data_reshape[2] + j] +=
                    input_[((i * data_reshape[1] + k) *
                        data_reshape[2] + j) * data_reshape[3] + t];
              }
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }
  }

  void Compute(int32_t *data_reshape,
               uint32_t data_reshape_size, uint32_t output_size) {
    base::memset(output_, static_cast<T>(0), output_size);
    switch (data_reshape_size) {
      case 1:Reduce1Dims(reduce_type_, data_reshape);
        break;
      case 2:Reduce2Dims(reduce_type_, data_reshape);
        break;
      case 3:Reduce3Dims(reduce_type_, data_reshape);
        break;
      case 4:Reduce4Dims(reduce_type_, data_reshape);
        break;
      default:LOG(FATAL) << "not implemented in mace"
                         << "data reshape size" << data_reshape_size
                         << "reduce first axis:" << reduce_first_axis_;
        break;
    }
  }

 private:
  const T *input_;
  const int32_t *input_dims_;
  uint32_t input_dim_size_;

  T *output_;

  bool reduce_first_axis_;
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_REDUCE_H_

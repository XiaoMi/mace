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

#include "mace/ops/reduce.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/core/tensor.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/reduce.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class ReduceOpBase : public Operation {
 public:
  explicit ReduceOpBase(OpConstructContext *context)
      : Operation(context),
        reduce_type_(
            static_cast<ReduceType>(Operation::GetOptionalArg<int>(
                "reduce_type", static_cast<int>(MEAN)))),
        axis_(Operation::GetRepeatedArgs<int>("axis")),
        keep_dims_(Operation::GetOptionalArg<bool>("keepdims", false)) {
  }

 protected:
  inline void Validate() {
    const Tensor *input = this->Input(0);
    const int left = static_cast<int>(input->dim_size() * -1);
    const int right = static_cast<int>(input->dim_size());
    if (axis_.size()) {
      for (unsigned int i = 0; i < axis_.size(); ++i) {
        MACE_CHECK(axis_[i] > left && axis_[i] < right, "Axis is over range.");
      }
    }
  }

 protected:
  ReduceType reduce_type_;
  std::vector<int> axis_;
  bool keep_dims_;
};

template<DeviceType D, class T>
class ReduceOp;

template<typename T>
class ReduceOp<DeviceType::CPU, T> : public ReduceOpBase {
 public:
  explicit ReduceOp(OpConstructContext *context)
      : ReduceOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Validate();
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    Simplify(input);
    // Use the same scale and zero point with input and output.
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    output->Resize(out_shape_);
    Compute(context, input, output);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void Simplify(const Tensor *input) {
    std::vector<bool> bitmap(static_cast<uint32_t>(input->dim_size()), false);
    if (axis_.empty()) {
      for (int i = 0; i < input->dim_size(); ++i) {
        bitmap[i] = true;
      }
    } else {
      for (unsigned int i = 0; i < axis_.size(); ++i) {
        int index = axis_[i] >= 0 ?
                    axis_[i] :
                    axis_[i] + input->dim_size();
        auto has_df = Operation::GetOptionalArg<int>(
            "has_data_format", 0);
        if (has_df && DataTypeToEnum<T>::value != DT_UINT8
            && input->dim_size() == 4) {
          if (index == 1 || index == 2) index = index + 1;
          else if (index == 3) index = 1;
        }
        bitmap[index] = true;
      }
    }
    out_shape_.clear();
    for (unsigned int i = 0; i < input->dim_size(); ++i) {
      if (!bitmap[i]) {
        out_shape_.push_back(input->dim(i));
      } else if (keep_dims_) {
        out_shape_.push_back(1);
      }
    }
    data_reshape_.clear();
    unsigned int dim_index = 0;
    for (; dim_index < input->dim_size(); ++dim_index) {
      if (input->dim(dim_index) != 1) break;
    }
    if (dim_index >= input->dim_size()) {
      reduce_first_axis_ = true;
    } else {
      reduce_first_axis_ = bitmap[dim_index];
      data_reshape_.push_back(input->dim(dim_index));
      ++dim_index;
      for (; dim_index < input->dim_size(); ++dim_index) {
        const int n = input->dim(dim_index);
        if (n == 1) {
          bitmap[dim_index] = bitmap[dim_index - 1];
        }
        if (bitmap[dim_index - 1] != bitmap[dim_index]) {
          data_reshape_.push_back(n);
        } else {
          data_reshape_.back() *= n;
        }
      }
    }
  }

  void Reduce1Dims(const OpContext *context,
                   const T *input,
                   ReduceType type,
                   T *output) {
    MACE_UNUSED(context);
    if (reduce_first_axis_) {
      if (type == ReduceType::MEAN) {
        T tmp = 0;
        for (int i = 0; i < data_reshape_[0]; ++i) {
          tmp = tmp + input[i];
        }
        output[0] = tmp / data_reshape_[0];
      } else if (type == ReduceType::MIN) {
        T tmp = input[0];
        for (int i = 1; i < data_reshape_[0]; ++i) {
          tmp = std::min<T>(tmp, input[i]);
        }
        output[0] = tmp;
      } else if (type == ReduceType::MAX) {
        T tmp = input[0];
        for (int i = 1; i < data_reshape_[0]; ++i) {
          tmp = std::max<T>(tmp, input[i]);
        }
        output[0] = tmp;
      } else if (type == ReduceType::PROD) {
        T tmp = input[0];
        for (int i = 1; i < data_reshape_[0]; ++i) {
          tmp = tmp * input[i];
        }
        output[0] = tmp;
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    } else {
      memcpy(output, input, data_reshape_[0] * sizeof(T));
    }
  }

  void Reduce2Dims(const OpContext *context,
                   const T *input,
                   ReduceType type,
                   T *output) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    if (reduce_first_axis_) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start; i < end; i += step) {
            T tmp = 0;
            for (int j = 0; j < data_reshape_[0]; ++j) {
              tmp += input[j * data_reshape_[1] + i];
            }
            output[i] = tmp / data_reshape_[0];
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i];
            for (int j = 1; j < data_reshape_[0]; ++j) {
              tmp = std::min(tmp, input[j * data_reshape_[1] + i]);
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i];
            for (int j = 1; j < data_reshape_[0]; ++j) {
              tmp = std::max(tmp, input[j * data_reshape_[1] + i]);
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i];
            for (int j = 1; j < data_reshape_[0]; ++j) {
              tmp = tmp * input[j * data_reshape_[1] + i];
            }
            output[i] = tmp;
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[1], 1);
    } else {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start; i < end; i += step) {
            T tmp = 0;
            for (int j = 0; j < data_reshape_[1]; ++j) {
              tmp += input[i * data_reshape_[1] + j];
            }
            output[i] = tmp / data_reshape_[1];
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i * data_reshape_[1]];
            for (int j = 1; j < data_reshape_[1]; ++j) {
              tmp = std::min(tmp, input[i * data_reshape_[1] + j]);
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i * data_reshape_[1]];
            for (int j = 1; j < data_reshape_[1]; ++j) {
              tmp = std::max(tmp, input[i * data_reshape_[1] + j]);
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i * data_reshape_[1]];
            for (int j = 1; j < data_reshape_[1]; ++j) {
              tmp = tmp * input[i * data_reshape_[1] + j];
            }
            output[i] = tmp;
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[0], 1);
    }
  }

  void Reduce3Dims(const OpContext *context,
                   const T *input,
                   ReduceType type,
                   T *output) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    if (reduce_first_axis_) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start; i < end; i += step) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                output[i] +=
                    input[(k * data_reshape_[1] + i) * data_reshape_[2]
                        + j];
              }
            }
            output[i] /= (data_reshape_[0] * data_reshape_[2]);
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i * data_reshape_[2]];
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                tmp = std::min(tmp,
                               input[
                                   (k * data_reshape_[1] + i) * data_reshape_[2]
                                       + j]);
              }
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start; i < end; i += step) {
            T tmp = input[i * data_reshape_[2]];
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                tmp =
                    std::max(tmp,
                             input[(k * data_reshape_[1] + i)
                                 * data_reshape_[2] + j]);
              }
            }
            output[i] = tmp;
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start; i < end; i += step) {
            T tmp = 1;
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                tmp *=
                    input[(k * data_reshape_[1] + i) * data_reshape_[2]
                        + j];
              }
            }
            output[i] = tmp;
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[1], 1);
    } else {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start; i < end; i += step) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                output[i * data_reshape_[2] + j] +=
                    input[(i * data_reshape_[1] + k) * data_reshape_[2]
                        + j];
              }
              output[i * data_reshape_[2] + j] /= data_reshape_[1];
            }
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start; i < end; i += step) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              T tmp = input[i * data_reshape_[1] * data_reshape_[2] + j];
              for (int k = 1; k < data_reshape_[1]; ++k) {
                tmp = std::min(tmp,
                               input[(i * data_reshape_[1] + k) *
                                   data_reshape_[2] + j]);
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start; i < end; i += step) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              T tmp = input[i * data_reshape_[1] * data_reshape_[2] + j];
              for (int k = 1; k < data_reshape_[1]; ++k) {
                tmp = std::max(tmp,
                               input[(i * data_reshape_[1] + k) *
                                   data_reshape_[2] + j]);
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start; i < end; i += step) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              T tmp = input[i * data_reshape_[1] * data_reshape_[2] + j];
              for (int k = 1; k < data_reshape_[1]; ++k) {
                tmp *= input[(i * data_reshape_[1] + k) *
                    data_reshape_[2] + j];
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[0], 1);
    }
  }

  void Reduce4Dims(const OpContext *context,
                   const T *input,
                   ReduceType type,
                   T *output) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    if (reduce_first_axis_) {
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  output[i * data_reshape_[3] + j] +=
                      input[((t * data_reshape_[1] + i) *
                          data_reshape_[2] + k) * data_reshape_[3] + j];
                }
              }
              output[i * data_reshape_[3] + j] /=
                  (data_reshape_[0] * data_reshape_[2]);
            }
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = input[i * data_reshape_[2] * data_reshape_[3] + j];
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  tmp = std::min(tmp,
                                 input[((t * data_reshape_[1] + i) *
                                     data_reshape_[2] + k) * data_reshape_[3]
                                     + j]);
                }
              }
              output[i * data_reshape_[3] + j] = tmp;
            }
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = input[i * data_reshape_[2] * data_reshape_[3] + j];
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  tmp = std::max(tmp,
                                 input[((t * data_reshape_[1] + i) *
                                     data_reshape_[2] + k) * data_reshape_[3]
                                     + j]);
                }
              }
              output[i * data_reshape_[3] + j] = tmp;
            }
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = 1;
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  tmp = tmp * input[((t * data_reshape_[1] + i) *
                      data_reshape_[2] + k) * data_reshape_[3] + j];
                }
              }
              output[i * data_reshape_[3] + j] = tmp;
            }
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[1], 1, 0, data_reshape_[3], 1);
    } else {
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        if (type == ReduceType::MEAN) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  output[i * data_reshape_[2] + j] +=
                      input[((i * data_reshape_[1] + k) *
                          data_reshape_[2] + j) * data_reshape_[3] + t];
                }
              }
              output[i * data_reshape_[2] + j] /=
                  (data_reshape_[1] * data_reshape_[3]);
            }
          }
        } else if (type == ReduceType::MIN) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = input[(i * data_reshape_[1] *
                  data_reshape_[2] + j) * data_reshape_[3]];
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  tmp =
                      std::min(tmp,
                               input[((i * data_reshape_[1] + k) *
                                   data_reshape_[2] + j) * data_reshape_[3]
                                   + t]);
                }
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else if (type == ReduceType::MAX) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = input[(i * data_reshape_[1] *
                  data_reshape_[2] + j) * data_reshape_[3]];
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  tmp =
                      std::max(tmp,
                               input[((i * data_reshape_[1] + k) *
                                   data_reshape_[2] + j) * data_reshape_[3]
                                   + t]);
                }
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else if (type == ReduceType::PROD) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t j = start1; j < end1; j += step1) {
              T tmp = 1;
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  tmp = tmp * input[((i * data_reshape_[1] + k) *
                      data_reshape_[2] + j) * data_reshape_[3] + t];
                }
              }
              output[i * data_reshape_[2] + j] = tmp;
            }
          }
        } else {
          MACE_NOT_IMPLEMENTED;
        }
      }, 0, data_reshape_[0], 1, 0, data_reshape_[2], 1);
    }
  }

  void Compute(const OpContext *context, const Tensor *input, Tensor *output) {
    Tensor::MappingGuard input_mapper(input);
    const T *input_ptr = input->data<T>();
    Tensor::MappingGuard output_map(output);
    T *output_ptr = output->mutable_data<T>();
    memset(output_ptr, 0, output->size() * sizeof(T));
    switch (data_reshape_.size()) {
      case 1:Reduce1Dims(context, input_ptr, reduce_type_, output_ptr);
        break;
      case 2:Reduce2Dims(context, input_ptr, reduce_type_, output_ptr);
        break;
      case 3:Reduce3Dims(context, input_ptr, reduce_type_, output_ptr);
        break;
      case 4:Reduce4Dims(context, input_ptr, reduce_type_, output_ptr);
        break;
      default:MACE_CHECK(false, "not implemented in mace")
          << "data reshape size" << data_reshape_.size()
          << "reduce first axis:" << reduce_first_axis_;
        break;
    }
  }

 private:
  bool reduce_first_axis_;
  std::vector<int> data_reshape_;
  std::vector<index_t> out_shape_;
};

#ifdef MACE_ENABLE_QUANTIZE
template<>
void ReduceOp<DeviceType::CPU, uint8_t>::Reduce1Dims(
    const OpContext *context,
    const uint8_t *input, ReduceType type, uint8_t *output) {
  MACE_UNUSED(context);
  if (reduce_first_axis_) {
    if (type == ReduceType::MEAN) {
      uint32_t tmp = 0;
      for (int i = 0; i < data_reshape_[0]; ++i) {
        tmp = tmp + input[i];
      }
      output[0] = static_cast<uint8_t>(
          (tmp + data_reshape_[0] / 2) / data_reshape_[0]);
    } else if (type == ReduceType::MIN) {
      uint8_t tmp = input[0];
      for (int i = 1; i < data_reshape_[0]; ++i) {
        tmp = std::min<uint8_t>(tmp, input[i]);
      }
      output[0] = tmp;
    } else if (type == ReduceType::MAX) {
      uint8_t tmp = input[0];
      for (int i = 1; i < data_reshape_[0]; ++i) {
        tmp = std::max<uint8_t>(tmp, input[i]);
      }
      output[0] = tmp;
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  } else {
    memcpy(output, input, data_reshape_[0] * sizeof(uint8_t));
  }
}

template<>
void ReduceOp<DeviceType::CPU, uint8_t>::Reduce2Dims(
    const OpContext *context,
    const uint8_t *input, ReduceType type, uint8_t *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  if (reduce_first_axis_) {
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start; i < end; i += step) {
          uint32_t tmp = 0;
          for (int j = 0; j < data_reshape_[0]; ++j) {
            tmp += input[j * data_reshape_[1] + i];
          }
          output[i] = static_cast<uint8_t>(
              (tmp + data_reshape_[0] / 2) / data_reshape_[0]);
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i];
          for (int j = 1; j < data_reshape_[0]; ++j) {
            tmp = std::min(tmp, input[j * data_reshape_[1] + i]);
          }
          output[i] = tmp;
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i];
          for (int j = 1; j < data_reshape_[0]; ++j) {
            tmp = std::max(tmp, input[j * data_reshape_[1] + i]);
          }
          output[i] = tmp;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[1], 1);
  } else {
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start; i < end; i += step) {
          uint32_t tmp = 0;
          for (int j = 0; j < data_reshape_[1]; ++j) {
            tmp += input[i * data_reshape_[1] + j];
          }
          output[i] = static_cast<uint8_t>(
              (tmp + data_reshape_[1] / 2) / data_reshape_[1]);
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i * data_reshape_[1]];
          for (int j = 1; j < data_reshape_[1]; ++j) {
            tmp = std::min(tmp, input[i * data_reshape_[1] + j]);
          }
          output[i] = tmp;
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i * data_reshape_[1]];
          for (int j = 1; j < data_reshape_[1]; ++j) {
            tmp = std::max(tmp, input[i * data_reshape_[1] + j]);
          }
          output[i] = tmp;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[0], 1);
  }
}

template<>
void ReduceOp<DeviceType::CPU, uint8_t>::Reduce3Dims(
    const OpContext *context,
    const uint8_t *input, ReduceType type, uint8_t *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  if (reduce_first_axis_) {
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start; i < end; i += step) {
          uint32_t tmp = 0;
          for (int j = 0; j < data_reshape_[2]; ++j) {
            for (int k = 0; k < data_reshape_[0]; ++k) {
              tmp += input[(k * data_reshape_[1] + i) * data_reshape_[2] + j];
            }
          }
          index_t dim = data_reshape_[0] * data_reshape_[2];
          output[i] = static_cast<uint8_t>((tmp + dim / 2) / dim);
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i * data_reshape_[2]];
          for (int j = 0; j < data_reshape_[2]; ++j) {
            for (int k = 0; k < data_reshape_[0]; ++k) {
              tmp = std::min(tmp,
                             input[(k * data_reshape_[1] + i) * data_reshape_[2]
                                 + j]);
            }
          }
          output[i] = tmp;
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start; i < end; i += step) {
          uint8_t tmp = input[i * data_reshape_[2]];
          for (int j = 0; j < data_reshape_[2]; ++j) {
            for (int k = 0; k < data_reshape_[0]; ++k) {
              tmp =
                  std::max(tmp,
                           input[(k * data_reshape_[1] + i)
                               * data_reshape_[2] + j]);
            }
          }
          output[i] = tmp;
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[1], 1);
  } else {
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint32_t tmp = 0;
            for (int k = 0; k < data_reshape_[1]; ++k) {
              tmp += input[(i * data_reshape_[1] + k) * data_reshape_[2] + j];
            }
            output[i * data_reshape_[2] + j] =
                static_cast<uint8_t>((tmp + data_reshape_[1] / 2) /
                    data_reshape_[1]);
          }
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[i * data_reshape_[1] * data_reshape_[2] + j];
            for (int k = 1; k < data_reshape_[1]; ++k) {
              tmp = std::min(tmp,
                             input[(i * data_reshape_[1] + k) *
                                 data_reshape_[2] + j]);
            }
            output[i * data_reshape_[2] + j] = tmp;
          }
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[i * data_reshape_[1] * data_reshape_[2] + j];
            for (int k = 1; k < data_reshape_[1]; ++k) {
              tmp = std::max(tmp,
                             input[(i * data_reshape_[1] + k) *
                                 data_reshape_[2] + j]);
            }
            output[i * data_reshape_[2] + j] = tmp;
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[0], 1, 0, data_reshape_[2], 1);
  }
}

template<>
void ReduceOp<DeviceType::CPU, uint8_t>::Reduce4Dims(
    const OpContext *context,
    const uint8_t *input, ReduceType type, uint8_t *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  if (reduce_first_axis_) {
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint32_t tmp = 0;
            for (int k = 0; k < data_reshape_[2]; ++k) {
              for (int t = 0; t < data_reshape_[0]; ++t) {
                tmp += input[((t * data_reshape_[1] + i) *
                    data_reshape_[2] + k) * data_reshape_[3] + j];
              }
            }
            index_t dim = data_reshape_[0] * data_reshape_[2];
            output[i * data_reshape_[3] + j] =
                static_cast<uint8_t>((tmp + dim / 2) / dim);
          }
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[i * data_reshape_[2] * data_reshape_[3] + j];
            for (int k = 0; k < data_reshape_[2]; ++k) {
              for (int t = 0; t < data_reshape_[0]; ++t) {
                tmp = std::min(tmp,
                               input[((t * data_reshape_[1] + i) *
                                   data_reshape_[2] + k) * data_reshape_[3]
                                   + j]);
              }
            }
            output[i * data_reshape_[3] + j] = tmp;
          }
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[i * data_reshape_[2] * data_reshape_[3] + j];
            for (int k = 0; k < data_reshape_[2]; ++k) {
              for (int t = 0; t < data_reshape_[0]; ++t) {
                tmp = std::max(tmp,
                               input[((t * data_reshape_[1] + i) *
                                   data_reshape_[2] + k) * data_reshape_[3]
                                   + j]);
              }
            }
            output[i * data_reshape_[3] + j] = tmp;
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[1], 1, 0, data_reshape_[3], 1);
  } else {
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      if (type == ReduceType::MEAN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint32_t tmp = 0;
            for (int k = 0; k < data_reshape_[1]; ++k) {
              for (int t = 0; t < data_reshape_[3]; ++t) {
                tmp += input[((i * data_reshape_[1] + k) *
                    data_reshape_[2] + j) * data_reshape_[3] + t];
              }
            }
            index_t dim = data_reshape_[1] * data_reshape_[3];
            output[i * data_reshape_[2] + j] =
                static_cast<uint8_t>((tmp + dim / 2) / dim);
          }
        }
      } else if (type == ReduceType::MIN) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[(i * data_reshape_[1] *
                data_reshape_[2] + j) * data_reshape_[3]];
            for (int k = 0; k < data_reshape_[1]; ++k) {
              for (int t = 0; t < data_reshape_[3]; ++t) {
                tmp =
                    std::min(tmp,
                             input[((i * data_reshape_[1] + k) *
                                 data_reshape_[2] + j) * data_reshape_[3] + t]);
              }
            }
            output[i * data_reshape_[2] + j] = tmp;
          }
        }
      } else if (type == ReduceType::MAX) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            uint8_t tmp = input[(i * data_reshape_[1] *
                data_reshape_[2] + j) * data_reshape_[3]];
            for (int k = 0; k < data_reshape_[1]; ++k) {
              for (int t = 0; t < data_reshape_[3]; ++t) {
                tmp =
                    std::max(tmp,
                             input[((i * data_reshape_[1] + k) *
                                 data_reshape_[2] + j) * data_reshape_[3] + t]);
              }
            }
            output[i * data_reshape_[2] + j] = tmp;
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }, 0, data_reshape_[0], 1, 0, data_reshape_[2], 1);
  }
}
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ReduceOp<DeviceType::GPU, T> : public ReduceOpBase {
 public:
  explicit ReduceOp(OpConstructContext *context)
      : ReduceOpBase(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ReduceKernel<T>>(reduce_type_,
                                                            axis_,
                                                            keep_dims_);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    Validate();
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLReduceKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterReduce(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Reduce", ReduceOp,
                   DeviceType::CPU, float);
#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Reduce", ReduceOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Reduce", ReduceOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Reduce", ReduceOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Reduce")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                bool keep_dims =
                    ProtoArgHelper::GetOptionalArg<OperatorDef, bool>(
                        *op, "keepdims", false);
                if (!keep_dims) {
                  return { DeviceType::CPU };
                }
                auto axis =
                    ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(
                        *op, "axis");
                if (axis.size() != 2 || axis[0] != 1 || axis[1] != 2) {
                  return { DeviceType::CPU };
                }
                auto tensor_shape_info = context->tensor_shape_info();
                if (tensor_shape_info->count(op->input(0)) == 0
                    || tensor_shape_info->at(op->input(0)).size() != 4) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace

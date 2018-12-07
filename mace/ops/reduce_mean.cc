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

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/reduce_mean.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

class ReduceMeanOpBase : public Operation {
 public:
  explicit ReduceMeanOpBase(OpConstructContext *context)
  : Operation(context),
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
  std::vector<int> axis_;
  bool keep_dims_;
};

template <DeviceType D, class T>
class ReduceMeanOp;

template <typename T>
class ReduceMeanOp<DeviceType::CPU, T> : public ReduceMeanOpBase {
 public:
  explicit ReduceMeanOp(OpConstructContext *context)
      : ReduceMeanOpBase(context) {
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Validate();
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    Simplify(input);
    output->Resize(out_shape_);
    Compute(input, output);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void Simplify(const Tensor *input) {
    std::vector<bool> bitmap(static_cast<uint32_t>(input->dim_size()), false);
    if (axis_.size() == 0) {
      for (int i = 0; i < input->dim_size(); ++i) {
        bitmap[i] = true;
      }
    } else {
      for (unsigned int i = 0; i < axis_.size(); ++i) {
        int index = axis_[i] >= 0 ?
                    axis_[i] :
                    axis_[i] + input->dim_size();
        // axis format is NHWC
        if (input->dim_size() == 4) {
          if (index == 1) index = 2;
          else if (index == 2) index = 3;
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
        if (bitmap[dim_index-1] != bitmap[dim_index]) {
          data_reshape_.push_back(n);
        } else {
          data_reshape_.back() *= n;
        }
      }
    }
  }

  void Compute(const Tensor *input, Tensor *output) {
    Tensor::MappingGuard input_mapper(input);
    const T *input_ptr = input->data<T>();
    Tensor::MappingGuard output_map(output);
    T *output_ptr = output->mutable_data<T>();
    memset(output_ptr, 0, output->size() * sizeof(T));
    switch (data_reshape_.size()) {
      case 1:
        if (reduce_first_axis_) {
          T sum = 0;
          for (int i = 0; i < data_reshape_[0]; ++i) {
            sum = sum + input_ptr[i];
          }
          output_ptr[0] = sum / data_reshape_[0];
        } else {
#pragma omp parallel for schedule(runtime)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            output_ptr[i] = input_ptr[i];
          }
        }
        break;
      case 2:
        if (reduce_first_axis_) {
#pragma omp parallel for schedule(runtime)
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[0]; ++j) {
              output_ptr[i] += input_ptr[j * data_reshape_[1] + i];
            }
            output_ptr[i] /= data_reshape_[0];
          }
        } else {
#pragma omp parallel for schedule(runtime)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[1]; ++j) {
              output_ptr[i] += input_ptr[i * data_reshape_[1] + j];
            }
            output_ptr[i] /= data_reshape_[1];
          }
        }
        break;
      case 3:
        if (reduce_first_axis_) {
#pragma omp parallel for schedule(runtime)
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[0]; ++k) {
                output_ptr[i] +=
                    input_ptr[(k * data_reshape_[1] + i) * data_reshape_[2]
                        + j];
              }
            }
            output_ptr[i] /= (data_reshape_[0] * data_reshape_[2]);
          }
        } else {
#pragma omp parallel for collapse(2) schedule(runtime)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                output_ptr[i * data_reshape_[2] + j] +=
                    input_ptr[(i * data_reshape_[1] + k) * data_reshape_[2]
                        + j];
              }
              output_ptr[i * data_reshape_[2] + j] /= data_reshape_[1];
            }
          }
        }
        break;
      case 4:
        if (reduce_first_axis_) {
#pragma omp parallel for collapse(2) schedule(runtime)
          for (int i = 0; i < data_reshape_[1]; ++i) {
            for (int j = 0; j < data_reshape_[3]; ++j) {
              for (int k = 0; k < data_reshape_[2]; ++k) {
                for (int t = 0; t < data_reshape_[0]; ++t) {
                  output_ptr[i * data_reshape_[3] + j] +=
                      input_ptr[((t * data_reshape_[1] + i) *
                          data_reshape_[2] + k)*data_reshape_[3] + j];
                }
              }
              output_ptr[i * data_reshape_[3] + j] /=
                  (data_reshape_[0] * data_reshape_[2]);
            }
          }
        } else {
#pragma omp parallel for collapse(2) schedule(runtime)
          for (int i = 0; i < data_reshape_[0]; ++i) {
            for (int j = 0; j < data_reshape_[2]; ++j) {
              for (int k = 0; k < data_reshape_[1]; ++k) {
                for (int t = 0; t < data_reshape_[3]; ++t) {
                  output_ptr[i * data_reshape_[2] + j] +=
                      input_ptr[((i * data_reshape_[1] + k) *
                          data_reshape_[2] + j)*data_reshape_[3] + t];
                }
              }
              output_ptr[i * data_reshape_[2] + j] /=
                  (data_reshape_[1] * data_reshape_[3]);
            }
          }
        }
        break;
      default:
        MACE_CHECK(false, "not implemented in mace")
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

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ReduceMeanOp<DeviceType::GPU, T> : public ReduceMeanOpBase {
 public:
  explicit ReduceMeanOp(OpConstructContext *context)
      : ReduceMeanOpBase(context) {
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::ReduceMeanKernel<T>(axis_, keep_dims_));
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
  std::unique_ptr<OpenCLReduceMeanKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterReduceMean(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ReduceMean", ReduceMeanOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "ReduceMean", ReduceMeanOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "ReduceMean", ReduceMeanOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

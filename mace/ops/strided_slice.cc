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

#include <algorithm>
#include <cmath>
#include <vector>

#include "mace/core/operator.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class StridedSliceOp : public Operation {
 public:
  explicit StridedSliceOp(OpConstructContext *context)
      : Operation(context),
        begin_mask_(Operation::GetOptionalArg<int>("begin_mask", 0)),
        end_mask_(Operation::GetOptionalArg<int>("end_mask", 0)),
        ellipsis_mask_(Operation::GetOptionalArg<int>("ellipsis_mask", 0)),
        new_axis_mask_(Operation::GetOptionalArg<int>("new_axis_mask", 0)),
        shrink_axis_mask_(
            Operation::GetOptionalArg<int>("shrink_axis_mask", 0)),
        is_slice_(Operation::GetOptionalArg<bool>("slice", false)),
        has_data_format_(Operation::GetOptionalArg<int>("has_data_format", 0)),
        checked_(false) {
    MACE_CHECK(ellipsis_mask_ == 0 && new_axis_mask_ == 0,
               "ellipsis_mask and new_axis_mask are not supported yet.");
  }

  void TransposeMaskValueFromNHWCToNCHW(int* mask_value) {
    size_t dims[4];
    int count;
    for (count = 0; count < 4; ++count) {
      dims[count] = *mask_value & 1;
      *mask_value >>= 1;
    }
    size_t new_dims[4] = {dims[0], dims[3], dims[1], dims[2]};
    for (count = 3; count >= 0; --count) {
      *mask_value <<= 1;
      *mask_value += new_dims[count];
    }
  }

  void TransposeDimsFromNHWCToNCHW(std::vector<int32_t>* dims) {
    int32_t h = (*dims)[1];
    int32_t w = (*dims)[2];
    int32_t c = (*dims)[3];

    (*dims)[1] = c;
    (*dims)[2] = h;
    (*dims)[3] = w;
  }

  void TransposeDimsFromNCHWToNHWC(std::vector<int32_t>* dims) {
    int32_t c = (*dims)[1];
    int32_t h = (*dims)[2];
    int32_t w = (*dims)[3];

    (*dims)[1] = h;
    (*dims)[2] = w;
    (*dims)[3] = c;
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);

    if (!checked_) {
      if (has_data_format_ && this->Input(0)->dim_size() == 4) {
        TransposeMaskValueFromNHWCToNCHW(&begin_mask_);
        TransposeMaskValueFromNHWCToNCHW(&end_mask_);
        TransposeMaskValueFromNHWCToNCHW(&ellipsis_mask_);
        TransposeMaskValueFromNHWCToNCHW(&new_axis_mask_);
        TransposeMaskValueFromNHWCToNCHW(&shrink_axis_mask_);
      }
      checked_ = true;
    }

    const Tensor *input = this->Input(INPUT);
    const Tensor *begin_indices = this->Input(BEGIN);
    const Tensor *end_indices = this->Input(END);
    const Tensor *strides = nullptr;

    if (this->InputSize() > 3) {
      strides = this->Input(STRIDES);
    }
    if (strides == nullptr) {
      tmp_strides_tensor_.Resize({begin_indices->size()});
      Tensor::MappingGuard strides_guard(&tmp_strides_tensor_);
      int32_t *strides_data = tmp_strides_tensor_.mutable_data<int32_t>();
      std::fill(strides_data, strides_data + tmp_strides_tensor_.size(), 1);
      strides = &tmp_strides_tensor_;
    }

    MACE_CHECK(begin_indices->dim_size() == 1 &&
               end_indices->dim_size() == 1 &&
               strides->dim_size() == 1,
               "Expected begin, end, and strides to be 1D tensor");

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard begin_indices_guard(begin_indices);
    Tensor::MappingGuard end_indices_guard(end_indices);
    Tensor::MappingGuard strides_guard(strides);
    const T *input_data = input->data<T>();
    const int32_t *begin_indices_data = begin_indices->data<int32_t>();
    const int32_t *end_indices_data = end_indices->data<int32_t>();
    const int32_t *strides_data = strides->data<int32_t>();

    std::vector<int32_t> begin_indices_vec(
        begin_indices_data, begin_indices_data + begin_indices->size());
    std::vector<int32_t> end_indices_vec(
        end_indices_data, end_indices_data + end_indices->size());
    std::vector<int32_t> strides_indices_vec(
        strides_data, strides_data + strides->size());

    MACE_CHECK(input->size() > 0 && input->dim_size() > 0 &&
               input->dim_size() <= 4,
               "The input size should larger than 0."
               " And input dims should be an integer in (0, 4].");

    std::vector<index_t> output_shape = {};

    const size_t input_dims = input->dim_size();
    if (is_slice_) {
      MACE_CHECK(begin_indices_vec.size() == input_dims &&
                 end_indices_vec.size() == input_dims,
                 "In slice, begin and size elements num should be equal");

      // transpose
      if (has_data_format_ && this->Input(0)->dim_size() == 4) {
        TransposeDimsFromNHWCToNCHW(&begin_indices_vec);
        TransposeDimsFromNHWCToNCHW(&end_indices_vec);
        TransposeDimsFromNHWCToNCHW(&strides_indices_vec);
      }

      for (size_t i = 0; i < input_dims; ++i) {
        if (end_indices_vec[i] == -1) {
          end_indices_vec[i] = input->dim(i) - begin_indices_vec[i];
        }
      }

      for (size_t i = 0; i < input_dims; ++i) {
        int32_t b = begin_indices_vec[i];
        int32_t s = end_indices_vec[i];
        int32_t input_i = input->dim(i);
        MACE_CHECK(0 <= b && b <= input_i,
                   "In Slice, expected begin[", i, "] in [0, ", input_i,
                   "], but got ", b);
        MACE_CHECK(0 <= s && b + s <= input_i,
                   "In Slice, expected size[", i, "] in [0, ",
                   input_i - b, "], but got", s);
        end_indices_vec[i] = b + s;
        output_shape.push_back(s);
      }
    } else {
      MACE_CHECK(begin_indices_vec.size() == end_indices_vec.size() &&
                 end_indices_vec.size() == strides_indices_vec.size(),
                 "In strided_slice, expected begin, end, and strides to be",
                 " equal size tensors");
      for (index_t i = 0; i < strides->size(); ++i) {
        MACE_CHECK(strides_indices_vec[i] != 0, "strides data cannot be 0!");
      }

      // pad
      begin_indices_vec.resize(input_dims, 0);
      strides_indices_vec.resize(input_dims, 1);
      std::vector<int32_t> tmp_input_dims(input->shape().begin(),
                                          input->shape().end());
      if (has_data_format_ && input_dims == 4) {
        TransposeDimsFromNCHWToNHWC(&tmp_input_dims);
      }
      for (size_t i = end_indices_vec.size(); i < input_dims; ++i) {
        end_indices_vec.push_back(tmp_input_dims[i]);
      }

      // transpose
      if (has_data_format_ && this->Input(0)->dim_size() == 4) {
        TransposeDimsFromNHWCToNCHW(&begin_indices_vec);
        TransposeDimsFromNHWCToNCHW(&end_indices_vec);
        TransposeDimsFromNHWCToNCHW(&strides_indices_vec);
      }

      // mask and shrink
      for (index_t d = 0; d < input->dim_size(); ++d) {
        index_t dim_len = input->dim(d);
        const std::vector<index_t> valid_range = {
            strides_indices_vec[d] > 0 ? 0 : -1,
            strides_indices_vec[d] > 0 ? dim_len : dim_len - 1};

        auto format_indices = [valid_range, dim_len](index_t indice) {
          index_t forward = indice < 0 ? indice + dim_len : indice;
          return Clamp(forward, valid_range[0], valid_range[1]);
        };

        if (!(shrink_axis_mask_ & (1 << d))) {
          if (begin_mask_ & (1 << d)) {
            begin_indices_vec[d] = strides_indices_vec[d] > 0 ? 0 : dim_len - 1;
          } else {
            begin_indices_vec[d] = format_indices(begin_indices_vec[d]);
          }
          if (end_mask_ & (1 << d)) {
            end_indices_vec[d] = strides_indices_vec[d] > 0 ? dim_len : -1;
          } else {
            end_indices_vec[d] = format_indices(end_indices_vec[d]);
          }

          int32_t out_dim_len = std::max(
              0.f, std::ceil((end_indices_vec[d] - begin_indices_vec[d]) /
                  static_cast<float>(strides_indices_vec[d])));
          output_shape.push_back(out_dim_len);
        } else {
          begin_indices_vec[d] = begin_indices_vec[d] < 0
                                      ? begin_indices_vec[d] + dim_len
                                      : begin_indices_vec[d];
          end_indices_vec[d] = begin_indices_vec[d] + 1;
          MACE_CHECK(
              begin_indices_vec[d] >= 0 && begin_indices_vec[d] < dim_len,
              "slice begin indice of dimension '", d, "': ",
              begin_indices_vec[d], ", is out of bound");
        }
      }
    }

    for (size_t i = 0; i < output_shape.size(); ++i) {
      MACE_CHECK(output_shape[i] > 0,
                 "Expected output_shape[", i, "] larger than 0, but got ",
                 output_shape[i]);
    }

    std::vector<index_t> dim_stride(input->dim_size(), 1);
    for (index_t d = input->dim_size() - 2; d >= 0; --d) {
      dim_stride[d] = dim_stride[d + 1] * input->dim(d + 1);
    }

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard output_guard(output);
    T *output_data = output->mutable_data<T>();

    bool slice_by_first_axis = true;
    if (strides_indices_vec[0] != 1) {
      slice_by_first_axis = false;
    } else {
      for (index_t d = 1; d < input->dim_size(); ++d) {
        if (strides_indices_vec[d] != 1 || begin_indices_vec[d] != 0 ||
            end_indices_vec[d] != input->dim(d)) {
          slice_by_first_axis = false;
          break;
        }
      }
    }

    if (slice_by_first_axis) {
      memcpy(output_data, input_data + begin_indices_vec[0] * dim_stride[0],
             sizeof(T) * (end_indices_vec[0] - begin_indices_vec[0]) *
                 dim_stride[0]);
    } else {
      if (input->dim_size() == 1) {
        for (index_t i = begin_indices_vec[0];
             strides_indices_vec[0] > 0 ? i < end_indices_vec[0]
                                 : i > end_indices_vec[0];
             i += strides_indices_vec[0]) {
          *output_data++ = input_data[i];
        }
      } else if (input->dim_size() == 2) {
        for (index_t i = begin_indices_vec[0];
             strides_indices_vec[0] > 0 ? i < end_indices_vec[0]
                                 : i > end_indices_vec[0];
             i += strides_indices_vec[0]) {
          for (index_t j = begin_indices_vec[1];
               strides_indices_vec[1] > 0 ? j < end_indices_vec[1]
                                   : j > end_indices_vec[1];
               j += strides_indices_vec[1]) {
            *output_data++ = input_data[i * input->dim(1) + j];
          }
        }
      } else if (input->dim_size() == 3) {
        for (index_t i = begin_indices_vec[0];
             strides_indices_vec[0] > 0 ? i < end_indices_vec[0]
                                 : i > end_indices_vec[0];
             i += strides_indices_vec[0]) {
          for (index_t j = begin_indices_vec[1];
               strides_indices_vec[1] > 0 ? j < end_indices_vec[1]
                                   : j > end_indices_vec[1];
               j += strides_indices_vec[1]) {
            for (index_t k = begin_indices_vec[2];
                 strides_indices_vec[2] > 0 ? k < end_indices_vec[2]
                                     : k > end_indices_vec[2];
                 k += strides_indices_vec[2]) {
              *output_data++ =
                  input_data[(i * input->dim(1) + j) * input->dim(2) + k];
            }
          }
        }
      } else if (input->dim_size() == 4) {
        for (index_t i = begin_indices_vec[0];
             strides_indices_vec[0] > 0 ? i < end_indices_vec[0]
                                 : i > end_indices_vec[0];
             i += strides_indices_vec[0]) {
          for (index_t j = begin_indices_vec[1];
               strides_indices_vec[1] > 0 ? j < end_indices_vec[1]
                                   : j > end_indices_vec[1];
               j += strides_indices_vec[1]) {
            for (index_t k = begin_indices_vec[2];
                 strides_indices_vec[2] > 0 ? k < end_indices_vec[2]
                                     : k > end_indices_vec[2];
                 k += strides_indices_vec[2]) {
              for (index_t l = begin_indices_vec[3];
                   strides_indices_vec[3] > 0 ? l < end_indices_vec[3]
                                       : l > end_indices_vec[3];
                   l += strides_indices_vec[3]) {
                *output_data++ =
                    input_data[((i * input->dim(1) + j) * input->dim(2) + k)
                               * input->dim(3) + l];
              }
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int begin_mask_;
  int end_mask_;
  int ellipsis_mask_;
  int new_axis_mask_;
  int shrink_axis_mask_;
  bool is_slice_;
  int has_data_format_;
  bool checked_;
  Tensor tmp_strides_tensor_;

  MACE_OP_INPUT_TAGS(INPUT, BEGIN, END, STRIDES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterStridedSlice(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "StridedSlice", StridedSliceOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "StridedSlice", StridedSliceOp,
                   DeviceType::CPU, int32_t);
}

}  // namespace ops
}  // namespace mace

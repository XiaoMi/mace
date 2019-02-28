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

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);

    auto df = static_cast<DataFormat>(Operation::GetOptionalArg<int>(
          "data_format", DataFormat::DF_NONE));

    if (!checked_) {
      if (df == DataFormat::NHWC && this->Input(0)->dim_size() == 4) {
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
    Tensor *output = this->Output(OUTPUT);
    if (strides == nullptr) {
      tmp_strides_tensor_.Resize({begin_indices->size()});
      Tensor::MappingGuard strides_guard(&tmp_strides_tensor_);
      int32_t *strides_data = tmp_strides_tensor_.mutable_data<int32_t>();
      std::fill(strides_data, strides_data + tmp_strides_tensor_.size(), 1);
      strides = &tmp_strides_tensor_;
    }

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard begin_indices_guard(begin_indices);
    Tensor::MappingGuard end_indices_guard(end_indices);
    Tensor::MappingGuard strides_guard(strides);
    const T *input_data = input->data<T>();
    const int32_t *begin_indices_data = begin_indices->data<int32_t>();
    const int32_t *end_indices_data = end_indices->data<int32_t>();
    const int32_t *strides_data = strides->data<int32_t>();
    std::vector<int32_t> pad_begin_indices(input->dim_size(), 0);
    std::vector<int32_t> pad_end_indices(input->dim_size(), 0);
    std::vector<int32_t> pad_strides_indices(input->dim_size(), 1);

    if (begin_indices->size() < input->dim_size()) {
      for (index_t i = 0; i < begin_indices->size(); ++i) {
        pad_begin_indices[i] = begin_indices_data[i];
        pad_end_indices[i] = end_indices_data[i];
        pad_strides_indices[i] = strides_data[i];
      }
      for (index_t i = begin_indices->size(); i < input->dim_size(); ++i) {
        pad_end_indices[i] = input->dim(i);
      }

      begin_indices_data = pad_begin_indices.data();
      end_indices_data = pad_end_indices.data();
      strides_data = pad_strides_indices.data();
    }

    std::vector<int32_t> transpose_begin_indices(input->dim_size(), 0);
    std::vector<int32_t> transpose_end_indices(input->dim_size(), 0);
    std::vector<int32_t> transpose_strides_indices(input->dim_size(), 1);
    if (df == DataFormat::NHWC && this->Input(0)->dim_size() == 4) {
      for (index_t i = 0; i < begin_indices->size(); ++i) {
        transpose_begin_indices[i] = begin_indices_data[i];
        transpose_end_indices[i] = end_indices_data[i];
        transpose_strides_indices[i] = strides_data[i];
      }
      TransposeDimsFromNHWCToNCHW(&transpose_begin_indices);
      TransposeDimsFromNHWCToNCHW(&transpose_end_indices);
      TransposeDimsFromNHWCToNCHW(&transpose_strides_indices);

      begin_indices_data = transpose_begin_indices.data();
      end_indices_data = transpose_end_indices.data();
      strides_data = transpose_strides_indices.data();
    }

    std::vector<int32_t> slice_end_data;
    if (is_slice_) {
      // if this op is slice, the end_indices_data is size actually
      slice_end_data.resize(end_indices->size());
      for (size_t i = 0; i < slice_end_data.size(); ++i) {
        if (end_indices_data[i] == -1) {
          slice_end_data[i] = input->dim(i);
        } else {
          slice_end_data[i] = begin_indices_data[i] + end_indices_data[i];
        }
      }
      end_indices_data = slice_end_data.data();
    }

    std::vector<index_t> output_shape;
    std::vector<index_t> real_begin_indices(input->dim_size(), 0);
    std::vector<index_t> real_end_indices(input->dim_size(), 0);
    for (index_t d = 0; d < input->dim_size(); ++d) {
      index_t dim_len = input->dim(d);
      if (begin_mask_ & (1 << d)) {
        real_begin_indices[d] = strides_data[d] > 0 ? 0 : dim_len - 1;
      } else {
        real_begin_indices[d] = (begin_indices_data[d] + dim_len) % dim_len;
      }
      if (end_mask_ & (1 << d)) {
        real_end_indices[d] = strides_data[d] > 0 ? dim_len : -1;
      } else {
        real_end_indices[d] =
            end_indices_data[d] < -dim_len
            ? -1
            : (end_indices_data[d] < 0
               ? (end_indices_data[d] + dim_len)
               : std::min(static_cast<index_t>(end_indices_data[d]),
                          dim_len));
      }

      int32_t out_dim_len = std::max(
          0.f, std::ceil((real_end_indices[d] - real_begin_indices[d]) /
              static_cast<float>(strides_data[d])));
      if (!(shrink_axis_mask_ & (1 << d))) {
        output_shape.push_back(out_dim_len);
      } else {
        MACE_CHECK(out_dim_len == 1,
                   "cannot shrink axis that has len > 1, dim(", d, "): [",
                   real_begin_indices[d], ", ", real_end_indices[d], "]");
      }
    }

    std::vector<index_t> dim_stride(input->dim_size(), 1);
    for (index_t d = input->dim_size() - 2; d >= 0; --d) {
      dim_stride[d] = dim_stride[d + 1] * input->dim(d + 1);
    }

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard output_guard(output);
    T *output_data = output->mutable_data<T>();

    bool slice_by_first_axis = true;
    if (strides_data[0] != 1) {
      slice_by_first_axis = false;
    } else {
      for (index_t d = 1; d < input->dim_size(); ++d) {
        if (strides_data[d] != 1 || real_begin_indices[d] != 0 ||
            real_end_indices[d] != input->dim(d)) {
          slice_by_first_axis = false;
          break;
        }
      }
    }

    if (slice_by_first_axis) {
      memcpy(output_data, input_data + real_begin_indices[0] * dim_stride[0],
             sizeof(T) * (real_end_indices[0] - real_begin_indices[0]) *
                 dim_stride[0]);
    } else {
      if (input->dim_size() == 1) {
        for (index_t i = real_begin_indices[0];
             strides_data[0] > 0 ? i < real_end_indices[0]
                                 : i > real_end_indices[0];
             i += strides_data[0]) {
          *output_data++ = input_data[i];
        }
      } else if (input->dim_size() == 2) {
        for (index_t i = real_begin_indices[0];
             strides_data[0] > 0 ? i < real_end_indices[0]
                                 : i > real_end_indices[0];
             i += strides_data[0]) {
          for (index_t j = real_begin_indices[1];
               strides_data[1] > 0 ? j < real_end_indices[1]
                                   : j > real_end_indices[1];
               j += strides_data[1]) {
            *output_data++ = input_data[i * input->dim(1) + j];
          }
        }
      } else if (input->dim_size() == 3) {
        for (index_t i = real_begin_indices[0];
             strides_data[0] > 0 ? i < real_end_indices[0]
                                 : i > real_end_indices[0];
             i += strides_data[0]) {
          for (index_t j = real_begin_indices[1];
               strides_data[1] > 0 ? j < real_end_indices[1]
                                   : j > real_end_indices[1];
               j += strides_data[1]) {
            for (index_t k = real_begin_indices[2];
                 strides_data[2] > 0 ? k < real_end_indices[2]
                                     : k > real_end_indices[2];
                 k += strides_data[2]) {
              *output_data++ =
                  input_data[(i * input->dim(1) + j) * input->dim(2) + k];
            }
          }
        }
      } else if (input->dim_size() == 4) {
        for (index_t i = real_begin_indices[0];
             strides_data[0] > 0 ? i < real_end_indices[0]
                                 : i > real_end_indices[0];
             i += strides_data[0]) {
          for (index_t j = real_begin_indices[1];
               strides_data[1] > 0 ? j < real_end_indices[1]
                                   : j > real_end_indices[1];
               j += strides_data[1]) {
            for (index_t k = real_begin_indices[2];
                 strides_data[2] > 0 ? k < real_end_indices[2]
                                     : k > real_end_indices[2];
                 k += strides_data[2]) {
              for (index_t l = real_begin_indices[3];
                   strides_data[3] > 0 ? l < real_end_indices[3]
                                       : l > real_end_indices[3];
                   l += strides_data[3]) {
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

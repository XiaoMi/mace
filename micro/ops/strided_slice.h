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

#ifndef MICRO_OPS_STRIDED_SLICE_H_
#define MICRO_OPS_STRIDED_SLICE_H_

#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/include/utils/macros.h"

namespace micro {
namespace ops {
template<typename T>
class StridedSliceOp : public framework::Operator {
 public:
  MaceStatus OnInit() {
    MACE_RETURN_IF_ERROR(InitPrams());

    return MACE_SUCCESS;
  }

  MaceStatus Run() {
    MACE_RETURN_IF_ERROR(AdjustPrams());
    MACE_RETURN_IF_ERROR(
        ResizeOutputShape(OUTPUT, output_shape_idx_, output_shape_));

    bool slice_by_first_axis = true;
    if (strides_[0] != 1) {
      slice_by_first_axis = false;
    } else {
      for (uint32_t d = 1; d < input_dim_size_; ++d) {
        if (strides_[d] != 1 || begin_[d] != 0 ||
            end_[d] != input_dims_[d]) {
          slice_by_first_axis = false;
          break;
        }
      }
    }

    if (slice_by_first_axis) {
      base::memset(dim_stride_, static_cast<int32_t>(1), input_dim_size_);
      for (int32_t d = input_dim_size_ - 2; d >= 0; --d) {
        dim_stride_[d] = dim_stride_[d + 1] * input_dims_[d + 1];
      }
      base::memcpy(output_, input_ + begin_[0] * dim_stride_[0],
                   sizeof(T) * (end_[0] - begin_[0]) * dim_stride_[0]);
    } else {
      if (input_dim_size_ == 1) {
        for (int32_t i = begin_[0];
             strides_[0] > 0 ? i < end_[0] : i > end_[0]; i += strides_[0]) {
          *output_++ = input_[i];
        }
      } else if (input_dim_size_ == 2) {
        for (int32_t i = begin_[0];
             strides_[0] > 0 ? i < end_[0] : i > end_[0]; i += strides_[0]) {
          for (int32_t j = begin_[1];
               strides_[1] > 0 ? j < end_[1] : j > end_[1]; j += strides_[1]) {
            *output_++ = input_[i * input_dims_[1] + j];
          }
        }
      } else if (input_dim_size_ == 3) {
        for (int32_t i = begin_[0];
             strides_[0] > 0 ? i < end_[0] : i > end_[0]; i += strides_[0]) {
          for (int32_t j = begin_[1];
               strides_[1] > 0 ? j < end_[1] : j > end_[1]; j += strides_[1]) {
            for (int32_t k = begin_[2];
                 strides_[2] > 0 ? k < end_[2] : k > end_[2];
                 k += strides_[2]) {
              *output_++ =
                  input_[(i * input_dims_[1] + j) * input_dims_[2] + k];
            }
          }
        }
      } else if (input_dim_size_ == 4) {
        for (int32_t i = begin_[0];
             strides_[0] > 0 ? i < end_[0] : i > end_[0]; i += strides_[0]) {
          for (int32_t j = begin_[1];
               strides_[1] > 0 ? j < end_[1] : j > end_[1]; j += strides_[1]) {
            for (int32_t k = begin_[2];
                 strides_[2] > 0 ? k < end_[2] : k > end_[2];
                 k += strides_[2]) {
              for (int32_t l = begin_[3];
                   strides_[3] > 0 ? l < end_[3] : l > end_[3];
                   l += strides_[3]) {
                int32_t input_base =
                    (i * input_dims_[1] + j) * input_dims_[2] + k;
                int32_t input_idx = input_base * input_dims_[3] + l;
                *output_++ = input_[input_idx];
              }
            }
          }
        }
      } else {
        MACE_NOT_IMPLEMENTED;
      }
    }
    return MACE_SUCCESS;
  }

 private:
  MaceStatus InitPrams() {
    input_ = GetInputData<T>(INPUT);
    input_dims_ = GetInputShapeDims(INPUT);
    input_dim_size_ = GetInputShapeDimSize(INPUT);
    MACE_ASSERT1(input_dim_size_ > 0 && input_dim_size_ <= 4,
                 "The input dims should be an integer in (0, 4].");

    ScratchBuffer scratch_buffer(engine_config_);
    begin_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    end_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    strides_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    output_shape_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    dim_stride_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);
    base::memset(begin_, static_cast<int32_t>(0), input_dim_size_);
    base::memset(end_, static_cast<int32_t>(0), input_dim_size_);
    base::memset(strides_, static_cast<int32_t>(1), input_dim_size_);

    begin_dims_ = GetInputShapeDims(BEGIN);
    end_dims_ = GetInputShapeDims(END);

    MACE_ASSERT1(
        GetInputShapeDimSize(BEGIN) == 1 && GetInputShapeDimSize(END) == 1,
        "Expected begin, end, and to be 1D tensor");

    output_ = GetOutputData<T>(OUTPUT);

    begin_mask_ = GetArgByName("begin_mask", static_cast<int32_t>(0));
    end_mask_ = GetArgByName("end_mask", static_cast<int32_t>(0));
    ellipsis_mask_ = GetArgByName("ellipsis_mask", static_cast<int32_t>(0));
    new_axis_mask_ = GetArgByName("new_axis_mask", static_cast<int32_t>(0));
    shrink_axis_mask_ =
        GetArgByName("shrink_axis_mask", static_cast<int32_t>(0));
    is_slice_ = GetArgByName("slice", false);
    MACE_ASSERT1(ellipsis_mask_ == 0 && new_axis_mask_ == 0,
                 "ellipsis_mask and new_axis_mask are not supported yet.");

    return MACE_SUCCESS;
  }

  int32_t FormatIndices(const int32_t (&valid_range)[2],
                        const int32_t dim_len, int32_t indice) {
    int32_t forward = indice < 0 ? indice + dim_len : indice;
    return base::clamp(forward, valid_range[0], valid_range[1]);
  }

  MaceStatus AdjustPrams() {
    const int32_t *begin = GetInputData<int32_t>(BEGIN);
    base::memcpy(begin_, begin, begin_dims_[0] * sizeof(int32_t));
    const int32_t *end = GetInputData<int32_t>(END);
    base::memcpy(end_, end, end_dims_[0] * sizeof(int32_t));

    const int32_t *strides = NULL;
    if (GetInputSize() > 3) {
      strides = GetInputData<int32_t>(STRIDES);
      strides_dims_ = GetInputShapeDims(STRIDES);
    }
    if (strides == NULL) {
      base::memset(strides_, static_cast<int32_t>(1), input_dim_size_);
      strides_dims_ = begin_dims_;
    } else {
      base::memcpy(strides_, strides, strides_dims_[0] * sizeof(int32_t));
    }

    output_shape_idx_ = 0;
    const uint32_t begin_size = static_cast<uint32_t>(begin_dims_[0]);
    MACE_UNUSED(begin_size);
    const uint32_t end_size = static_cast<uint32_t>(end_dims_[0]);
    if (is_slice_) {
      MACE_ASSERT1(begin_size == input_dim_size_ && end_size == input_dim_size_,
                   "In slice, begin and size elements num should be equal");
      for (uint32_t i = 0; i < input_dim_size_; ++i) {
        if (end_[i] == -1) {
          end_[i] = input_dims_[i] - begin_[i];
        }
      }
      for (uint32_t i = 0; i < input_dim_size_; ++i) {
        int32_t b = begin_[i];
        int32_t s = end_[i];
#ifndef NDEBUG
        int32_t input_i = input_dims_[i];
        if (!(0 <= b && b <= input_i)) {
          LOG(FATAL) << "In Slice, expected begin[" << i << "] in [0, "
                     << input_i << "], but got " << b;
        }
        if (!(0 <= s && b + s <= input_i)) {
          LOG(FATAL) << "In Slice, expected size[" << i << "] in [0, "
                     << input_i - b << "], but got" << s;
        }
#endif
        end_[i] = b + s;
        output_shape_[output_shape_idx_++] = s;
      }
    } else {
      const uint32_t strides_size = static_cast<uint32_t>(strides_dims_[0]);
      MACE_ASSERT2(begin_size == end_size && end_size == strides_size,
                   "In strided_slice, expected begin, end, and strides to be",
                   " equal size tensors");
      for (uint32_t i = 0; i < strides_size; ++i) {
        MACE_ASSERT1(strides_[i] != 0, "strides data cannot be 0!");
      }

      // pad
      for (uint32_t i = end_size; i < input_dim_size_; ++i) {
        end_[i] = input_dims_[i];
      }

      // mask and shrink
      for (uint32_t d = 0; d < input_dim_size_; ++d) {
        int32_t dim_len = input_dims_[d];
        const int32_t valid_range[] = {strides_[d] > 0 ? 0 : -1,
                                       strides_[d] > 0 ? dim_len : dim_len - 1};
        if (!(shrink_axis_mask_ & (1 << d))) {
          if (begin_mask_ & (1 << d)) {
            begin_[d] = strides_[d] > 0 ? 0 : dim_len - 1;
          } else {
            begin_[d] = FormatIndices(valid_range, dim_len, begin_[d]);
          }
          if (end_mask_ & (1 << d)) {
            end_[d] = strides_[d] > 0 ? dim_len : -1;
          } else {
            end_[d] = FormatIndices(valid_range, dim_len, end_[d]);
          }

          int32_t out_dim_len = base::max(
              static_cast<int32_t>(0), base::ceil((end_[d] - begin_[d]) /
                  static_cast<float>(strides_[d])));
          output_shape_[output_shape_idx_++] = out_dim_len;
        } else {
          begin_[d] = begin_[d] < 0 ? begin_[d] + dim_len : begin_[d];
          end_[d] = begin_[d] + 1;
#ifndef NDEBUG
          if (!(begin_[d] >= 0 && begin_[d] < dim_len)) {
            LOG(FATAL) << "slice begin indice of dimension '" << d << "': "
                       << begin_[d] << ", is out of bound";
          }
#endif
        }
      }
    }
#ifndef NDEBUG
    for (uint32_t i = 0; i < output_shape_idx_; ++i) {
      if (output_shape_[i] <= 0) {
        LOG(FATAL) << "Expected output_shape[" << i
                   << "] larger than 0, but got " << output_shape_[i];
      }
    }
#endif
    return MACE_SUCCESS;
  }

 private:
  const T *input_;
  const int32_t *input_dims_;
  uint32_t input_dim_size_;
  int32_t *begin_;
  const int32_t *begin_dims_;
  int32_t *end_;
  const int32_t *end_dims_;
  int32_t *strides_;
  const int32_t *strides_dims_;

  T *output_;
  int32_t *output_shape_;
  uint32_t output_shape_idx_;
  int32_t *dim_stride_;

  int32_t begin_mask_;
  int32_t end_mask_;
  int32_t ellipsis_mask_;
  int32_t new_axis_mask_;
  int32_t shrink_axis_mask_;
  bool is_slice_;

  MACE_OP_INPUT_TAGS(INPUT, BEGIN, END, STRIDES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro

#endif  // MICRO_OPS_STRIDED_SLICE_H_

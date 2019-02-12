// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#ifndef MACE_OPS_REF_CONV_2D_H_
#define MACE_OPS_REF_CONV_2D_H_

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"

namespace mace {
namespace ops {
namespace ref {

template<typename OUTPUT_TYPE>
class Conv2d {
 public:
  Conv2d(int stride_h, int stride_w, int dilation_h, int dilation_w);
  ~Conv2d() {}
  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output);
};

template<>
class Conv2d<float> {
 public:
  Conv2d(int pad_h,
         int pad_w,
         int stride_h,
         int stride_w,
         int dilation_h,
         int dilation_w)
      : pad_h_(pad_h),
        pad_w_(pad_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w) {}
  ~Conv2d() {}
  // Always row-major after transpose
  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output);

 private:
  int pad_h_;
  int pad_w_;
  int stride_h_;
  int stride_w_;
  int dilation_h_;
  int dilation_w_;
};

}  // namespace ref
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_REF_CONV_2D_H_


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

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ConvPool2dOpBase : public Operator<D, T> {
 public:
  ConvPool2dOpBase(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        strides_(OperatorBase::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(OperatorBase::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(OperatorBase::GetRepeatedArgs<int>("padding_values")),
        dilations_(OperatorBase::GetRepeatedArgs<int>("dilations", {1, 1})) {}

 protected:
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_POOL_2D_BASE_H_

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

#ifndef MICRO_MODEL_NET_DEF_H_
#define MICRO_MODEL_NET_DEF_H_

#include "micro/base/serialize.h"
#include "micro/model/argument.h"
#include "micro/model/const_tensor.h"
#include "micro/model/input_output_info.h"
#include "micro/model/operator_def.h"

namespace micro {
namespace model {

class NetDef : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(NetDef)

  MACE_DECLARE_PTR_ARRAY_FUNC(OperatorDef, op);
  MACE_DECLARE_PTR_ARRAY_FUNC(Argument, arg);
  MACE_DECLARE_PTR_ARRAY_FUNC(ConstTensor, tensor);
  MACE_DECLARE_OBJECT_FUNC(int32_t, data_type);
  MACE_DECLARE_PTR_ARRAY_FUNC(InputOutputInfo, input_info);
  MACE_DECLARE_PTR_ARRAY_FUNC(InputOutputInfo, output_info);

 private:
  SerialArray<OperatorDef> ops_;
  SerialArray<Argument> args_;
  SerialArray<ConstTensor> tensors_;
  SerialInt32 data_type_;
  SerialArray<InputOutputInfo> input_infos_;
  SerialArray<InputOutputInfo> output_infos_;
};

}  // namespace model
}  // namespace micro

#endif  // MICRO_MODEL_NET_DEF_H_

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

#include "micro/model/net_def.h"

namespace micro {
namespace model {

MACE_DEFINE_PTR_ARRAY_FUNC(NetDef, OperatorDef, op, ops_)

MACE_DEFINE_PTR_ARRAY_FUNC(NetDef, Argument, arg, args_)

MACE_DEFINE_PTR_ARRAY_FUNC(NetDef, ConstTensor, tensor, tensors_)

MACE_DEFINE_OBJECT_FUNC(NetDef, int32_t, data_type)

MACE_DEFINE_PTR_ARRAY_FUNC(NetDef, InputOutputInfo, input_info, input_infos_)

MACE_DEFINE_PTR_ARRAY_FUNC(NetDef, InputOutputInfo, output_info, output_infos_)

}  // namespace model
}  // namespace micro

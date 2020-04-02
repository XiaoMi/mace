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

#include "micro/model/operator_def.h"

namespace micro {
namespace model {

MACE_DEFINE_STRING_ARRAY_FUNC(OperatorDef, input, inputs_)
MACE_DEFINE_STRING_ARRAY_FUNC(OperatorDef, output, outputs_)
MACE_DEFINE_STRING_FUNC(OperatorDef, name, name_)
MACE_DEFINE_STRING_FUNC(OperatorDef, type, type_)
MACE_DEFINE_OBJECT_FUNC(OperatorDef, int32_t, device_type)
MACE_DEFINE_PTR_ARRAY_FUNC(OperatorDef, Argument, arg, args_)
MACE_DEFINE_PTR_ARRAY_FUNC(OperatorDef, OutputShape,
                           output_shape, output_shapes_)
MACE_DEFINE_ARRAY_FUNC(OperatorDef, DataType, output_type, output_types_)
// the mem_offset is the mem_id in proto file
MACE_DEFINE_ARRAY_FUNC(OperatorDef, int32_t, mem_offset, mem_offsets_)

}  // namespace model
}  // namespace micro

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

#ifndef MICRO_MODEL_OPERATOR_DEF_H_
#define MICRO_MODEL_OPERATOR_DEF_H_

#include "micro/base/serialize.h"
#include "micro/include/public/micro.h"
#include "micro/model/argument.h"
#include "micro/model/output_shape.h"

namespace micro {
namespace model {

class OperatorDef : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(OperatorDef)

  MACE_DECLARE_STRING_ARRAY_FUNC(input);
  MACE_DECLARE_STRING_ARRAY_FUNC(output);
  MACE_DECLARE_STRING_FUNC(name);
  MACE_DECLARE_STRING_FUNC(type);
  MACE_DECLARE_OBJECT_FUNC(int32_t, device_type);
  MACE_DECLARE_PTR_ARRAY_FUNC(Argument, arg);
  MACE_DECLARE_PTR_ARRAY_FUNC(OutputShape, output_shape);
  MACE_DECLARE_ARRAY_FUNC(DataType, output_type);
  // the mem_offset is the mem_id in proto file
  MACE_DECLARE_ARRAY_FUNC(int32_t, mem_offset);

 private:
  SerialArray<SerialString> inputs_;
  SerialArray<SerialString> outputs_;
  SerialString name_;
  SerialString type_;
  // device_type_ is not used currently, for future;
  SerialInt32 device_type_;
  SerialArray<Argument> args_;
  SerialArray<OutputShape> output_shapes_;
  SerialArray<DataType> output_types_;
  SerialArray<SerialInt32> mem_offsets_;
};

}  // namespace model
}  // namespace micro

#endif  // MICRO_MODEL_OPERATOR_DEF_H_

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

#ifndef MICRO_MODEL_INPUT_OUTPUT_INFO_H_
#define MICRO_MODEL_INPUT_OUTPUT_INFO_H_

#include "micro/base/serialize.h"

namespace micro {
namespace model {

class InputOutputInfo : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(InputOutputInfo)

  MACE_DECLARE_STRING_FUNC(name);
  MACE_DECLARE_OBJECT_FUNC(int32_t, node_id);
  MACE_DECLARE_ARRAY_FUNC(int32_t, dim);
  MACE_DECLARE_OBJECT_FUNC(int32_t, max_byte_size);
  MACE_DECLARE_OBJECT_FUNC(int32_t, data_type);
  MACE_DECLARE_OBJECT_FUNC(int32_t, data_format);
  MACE_DECLARE_OBJECT_FUNC(float, scale);
  MACE_DECLARE_OBJECT_FUNC(int32_t, zero_point);

 private:
  SerialString name_;
  SerialInt32 node_id_;
  SerialArray<SerialInt32> dims_;
  SerialInt32 max_byte_size_;
  SerialInt32 data_type_;
  SerialInt32 data_format_;
  SerialFloat scale_;
  SerialInt32 zero_point_;
};

}  // namespace model
}  // namespace micro

#endif  // MICRO_MODEL_INPUT_OUTPUT_INFO_H_

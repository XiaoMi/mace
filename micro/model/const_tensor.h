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

#ifndef MICRO_MODEL_CONST_TENSOR_H_
#define MICRO_MODEL_CONST_TENSOR_H_

#include "micro/base/serialize.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace model {

class ConstTensor : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(ConstTensor)

  MACE_DECLARE_ARRAY_FUNC(int32_t, dim);
  MACE_DECLARE_OBJECT_FUNC(DataType, data_type);
  MACE_DECLARE_ARRAY_FUNC(float, float_data);
  MACE_DECLARE_ARRAY_FUNC(int32_t, int32_data);
  MACE_DECLARE_STRING_FUNC(name);
  MACE_DECLARE_OBJECT_FUNC(int32_t, offset);
  MACE_DECLARE_OBJECT_FUNC(int32_t, data_size);
  MACE_DECLARE_OBJECT_FUNC(float, scale);
  MACE_DECLARE_OBJECT_FUNC(int32_t, zero_point);
  MACE_DECLARE_OBJECT_FUNC(float, minval);
  MACE_DECLARE_OBJECT_FUNC(float, maxval);
  MACE_DECLARE_OBJECT_FUNC(bool, quantized);
  MACE_DECLARE_OBJECT_FUNC(uint32_t, node_id);

  const int32_t *dim() const;

 private:
  SerialArray<SerialInt32> dims_;
  DataType data_type_;
  SerialArray<SerialFloat> float_datas_;
  SerialArray<SerialInt32> int32_datas_;
  SerialString name_;
  SerialInt32 offset_;
  SerialInt32 data_size_;
  SerialFloat scale_;
  SerialInt32 zero_point_;
  SerialFloat minval_;
  SerialFloat maxval_;
  SerialBool quantized_;
  SerialUint32 node_id_;
};

}  // namespace model
}  // namespace micro

#endif  // MICRO_MODEL_CONST_TENSOR_H_

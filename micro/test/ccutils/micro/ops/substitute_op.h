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

#ifndef MICRO_TEST_CCUTILS_MICRO_OPS_SUBSTITUTE_OP_H_
#define MICRO_TEST_CCUTILS_MICRO_OPS_SUBSTITUTE_OP_H_

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace framework {

const uint32_t kMaxInputNum = 10;
const uint32_t kMaxOutputNum = 4;
const uint32_t kMaxArgNum = 20;

struct Arg {
  const char *name;
  float value;
};

struct RepeatArg {
  const char *name;
  const void *ptr;
  uint32_t length;
};

class SubstituteOp {
 public:
  SubstituteOp();
  ~SubstituteOp() {}

  SubstituteOp &AddInput(const void *input,
                         const int32_t *dims, const uint32_t dims_size);
  SubstituteOp &AddOutput(void *output,
                          int32_t *dims, const uint32_t dims_size);

  template<typename T>
  SubstituteOp &AddArg(const char *name, T value) {
    MACE_ASSERT(arg_idx_ < kMaxArgNum);
    args_[arg_idx_].name = name;
    args_[arg_idx_].value = static_cast<float>(value);
    ++arg_idx_;
    return *this;
  }

  template<typename T>
  SubstituteOp &AddRepeatArg(const char *name, const T *value, uint32_t len) {
    MACE_ASSERT(repeat_arg_idx_ < kMaxArgNum);
    repeat_args_[repeat_arg_idx_].name = name;
    repeat_args_[repeat_arg_idx_].ptr = value;
    repeat_args_[repeat_arg_idx_].length = len;
    ++repeat_arg_idx_;
    return *this;
  }

 public:
  template<typename T>
  T GetArgByName(const char *name, T default_value) const {
    for (uint32_t i = 0; i < arg_idx_; ++i) {
      if (base::strcmp(name, args_[i].name) == 0) {
        return static_cast<T>(args_[i].value);
      }
    }
    return default_value;
  }

  template<typename T>
  const T *GetRepeatArgByName(
      const char *name, uint32_t *size = NULL) const {
    for (uint32_t i = 0; i < repeat_arg_idx_; ++i) {
      if (base::strcmp(name, repeat_args_[i].name) == 0) {
        if (size != NULL) {
          *size = repeat_args_[i].length;
        }
        return static_cast<const T *>(repeat_args_[i].ptr);
      }
    }
    if (size != NULL) {
      *size = 0;
    }
    return NULL;
  }

  uint32_t GetInputSize();
  const void *DoGetInputData(uint32_t idx);
  uint32_t GetInputShapeDimSize(uint32_t idx);
  const int32_t *GetInputShapeDims(uint32_t idx);
  uint32_t GetOutputSize();
  void *DoGetOutputData(uint32_t idx);
  uint32_t GetOutputShapeDimSize(uint32_t idx);
  const int32_t *GetOutputShapeDims(uint32_t idx);
  MaceStatus ResizeOutputShape(uint32_t idx, uint32_t input_dim_size,
                               const int32_t *input_dims);
  MaceStatus ReuseInputBufferForOutput(uint32_t output_idx, uint32_t input_idx);

  template<typename T>
  const T *GetInputData(uint32_t idx) {
    return static_cast<const T *>(DoGetInputData(idx));
  }

  template<typename T>
  T *GetOutputData(uint32_t idx) {
    return static_cast<T *>(DoGetOutputData(idx));
  }

 private:
  const void *inputs_[kMaxInputNum];
  const int32_t *input_dims_[kMaxInputNum];
  uint32_t input_dim_sizes_[kMaxInputNum];
  uint32_t input_idx_;

  void *outputs_[kMaxOutputNum];
  int32_t *output_dims_[kMaxOutputNum];
  uint32_t output_dim_sizes_[kMaxOutputNum];
  uint32_t output_idx_;

  // for arg
  Arg args_[kMaxArgNum];
  uint32_t arg_idx_;
  RepeatArg repeat_args_[kMaxArgNum];
  uint32_t repeat_arg_idx_;
};

}  // namespace framework
}  // namespace micro

#endif  // MICRO_TEST_CCUTILS_MICRO_OPS_SUBSTITUTE_OP_H_

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

#ifndef MICRO_FRAMEWORK_OPERATOR_H_
#define MICRO_FRAMEWORK_OPERATOR_H_

#include "micro/base/logging.h"
#include "micro/base/types.h"
#include "micro/include/public/micro.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {

struct MaceMicroEngineConfig;

namespace model {
class Argument;
class OperatorDef;
class OutputShape;
}  // namespace model

namespace ops {
typedef framework::ScratchBuffer ScratchBuffer;
}

namespace framework {

#ifndef MACE_OP_INPUT_TAGS
#define MACE_OP_INPUT_TAGS(first_input, ...) \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#endif  // MACE_OP_INPUT_TAGS

#ifndef MACE_OP_OUTPUT_TAGS
#define MACE_OP_OUTPUT_TAGS(first_input, ...) \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }
#endif  // MACE_OP_OUTPUT_TAGS

class OpContext;

class Operator {
 public:
  Operator() {}
  // Note: This func should be virtual, but if we make it virtual,
  // the operator delete will be needed, which is in c++ runtime library.
  // For we don't use the Operator pointer to point sub-classes, the
  // virtual ~Operator() is not needed.
  ~Operator();

  MaceStatus Init(MaceMicroEngineConfig *engine_config,
                  OpContext *op_context,
                  const model::OperatorDef *op_def);
  virtual MaceStatus OnInit();
  virtual MaceStatus Run();

  template<typename T>
  T GetArgByName(const char *name, T default_value) const;

  template<typename T>
  const T *GetRepeatArgByName(const char *name,
                              uint32_t *size = NULL) const;

 protected:
  uint32_t GetInputSize();
  const void *DoGetInputData(uint32_t idx);
  uint32_t GetInputShapeDimSize(uint32_t idx);
  const int32_t *GetInputShapeDims(uint32_t idx);
  uint32_t GetOutputSize();
  DataType GetOutputDataType(uint32_t idx);
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
  const model::Argument *GetArgByName(const char *name) const;

 protected:
  const model::OperatorDef *op_def_;
  MaceMicroEngineConfig *engine_config_;

 private:
  OpContext *op_context_;
};

}  // namespace framework
}  // namespace micro

#endif  // MICRO_FRAMEWORK_OPERATOR_H_

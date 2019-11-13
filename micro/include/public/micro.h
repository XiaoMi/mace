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

#ifndef MICRO_INCLUDE_PUBLIC_MICRO_H_
#define MICRO_INCLUDE_PUBLIC_MICRO_H_

#include <stdint.h>

#include "micro/include/port/define.h"

namespace micro {

enum DataFormat {
  NONE = 0, NHWC = 1, NCHW = 2,
  HWOI = 100, OIHW = 101, HWIO = 102, OHWI = 103,
  AUTO = 1000,
};

enum PerfHint {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
};

enum DataType {
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_UINT8 = 2,
  DT_HALF = 3,
  DT_INT32 = 4,
  DT_FLOAT16 = 5,
  DT_BFLOAT16 = 6,
};

enum MaceStatus {
  MACE_SUCCESS = 0,
  MACE_INVALID_ARGS = 1,
  MACE_OUT_OF_RESOURCES = 2,
  MACE_UNSUPPORTED = 3,
  MACE_RUNTIME_ERROR = 4,
};

namespace model {
class NetDef;
}  // namespace model

namespace framework {
class Graph;
class Operator;
}  // namespace framework

struct MACE_API MaceMicroEngineConfig {
  model::NetDef *net_def_;
  const uint8_t *model_data_;
  framework::Graph *graph_;
  framework::Operator **op_array_;
  uint8_t *tensor_mem_;
  const void **input_buffers_;
  const int32_t **input_shapes_;
  uint8_t *scratch_buffer_;
  uint32_t scratch_buffer_size_;
};

class MACE_API MaceMicroEngine {
 public:
  MaceMicroEngine() {}
  ~MaceMicroEngine() {}

  MaceStatus Init(MaceMicroEngineConfig *engine_config);

  MaceStatus RegisterInputData(uint32_t idx, const void *input_buffer,
                               const int32_t *input_dims);
  MaceStatus Run();

  MaceStatus GetOutputData(const uint32_t idx, void **output_data,
                           const int32_t **output_dims,
                           uint32_t *output_dim_size);
  MaceStatus GetOpOutputData(const uint32_t op_def_idx,
                             const uint32_t output_idx,
                             void **output_data,
                             const int32_t **output_dims,
                             uint32_t *output_dim_size);

 private:
  MaceMicroEngineConfig *engine_config_;

  MaceMicroEngine(const MaceMicroEngine &);
  MaceMicroEngine &operator=(const MaceMicroEngine &);
};

}  // namespace micro

#endif  // MICRO_INCLUDE_PUBLIC_MICRO_H_

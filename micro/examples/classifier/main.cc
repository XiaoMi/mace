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

#include <cstdio>

#include "data.h"
#include "micro.h"

namespace micro {
namespace MICRO_MODEL_NAME {

MaceStatus GetMicroEngineSingleton(MaceMicroEngine **engine);

}
}  // namespace micro

int main() {
  micro::MaceMicroEngine *micro_engine = NULL;
  micro::MICRO_MODEL_NAME::GetMicroEngineSingleton(&micro_engine);

  micro_engine->RegisterInputData(0, MICRO_DATA_NAME::input,
                                  MICRO_DATA_NAME::input_dims);
  micro_engine->Run();

  float *output_buffer = NULL;
  const int32_t *output_dims = NULL;
  uint32_t dim_size = 0;
  micro_engine->GetOutputData(0, reinterpret_cast<void **>(&output_buffer),
                              &output_dims, &dim_size);

  int32_t output_total_size = 1;
  for (int32_t i = 0; i < dim_size; ++i) {
    output_total_size *= output_dims[i];
  }

  for (int32_t i = 0; i < output_total_size; ++i) {
    printf("%d: %f\n", i, output_buffer[i]);
  }

  return 0;
}

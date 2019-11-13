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


#include <gtest/gtest.h>

#include "micro/base/logging.h"
#include "micro/include/public/micro.h"

#ifndef MICRO_MODEL_NAME
#error Please specify model name in the command
#endif

namespace micro {

namespace MICRO_MODEL_NAME {
MaceStatus GetMicroEngineSingleton(MaceMicroEngine **engine);
}  // namespace MICRO_MODEL_NAME

class EngineTest : public ::testing::Test {
};

void OutputAllInfo() {
  MaceMicroEngine *micro_engine = NULL;
  MACE_ASSERT(MICRO_MODEL_NAME::GetMicroEngineSingleton(&micro_engine)
                  == MACE_SUCCESS && micro_engine != NULL);

  float input_buffer[1 * 1 * 128 * 9] = {0};
  int32_t input_shape[] = {1, 1, 128, 9};

  micro_engine->RegisterInputData(0, input_buffer, input_shape);
  MACE_ASSERT(MACE_SUCCESS == micro_engine->Run());

  void *output_buffer = NULL;
  const int32_t *output_dims = NULL;
  uint32_t dim_size = 0;
  micro_engine->GetOutputData(0, &output_buffer, &output_dims, &dim_size);
  LOG(INFO) << "EngineTest success, dim_size=" << dim_size;
}

TEST_F(EngineTest, OutputAllInfo) {
  OutputAllInfo();
}

}  // namespace micro

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

#include <HAP_farf.h>

#include "micro/include/public/micro.h"
#include "rpc/skel/base_func.h"

#ifndef MICRO_MODEL_NAME
#error Please specify model name in the command
#endif

namespace micro {

namespace MICRO_MODEL_NAME {
MaceStatus GetMicroEngineSingleton(MaceMicroEngine **engine);
}  // namespace MICRO_MODEL_NAME

namespace port {
namespace api {
int64_t NowMicros();
}  // namespace api
}  // namespace port

namespace testing {

namespace {
const int32_t kMicroRunTestTimes = 10;
const int32_t input0_shape[4] = {1, 1, 128, 9};
const int32_t input_length = 1 * 1 * 128 * 9;
float input0[input_length] = {0};
}  // namespace

void MicroRunModel() {
  int64_t t0 = port::api::NowMicros();
  MaceMicroEngine *micro_engine = NULL;
  MICRO_MODEL_NAME::GetMicroEngineSingleton(&micro_engine);
  int64_t t1 = port::api::NowMicros();
  double init_millis = (t1 - t0) / 1000.0;
  FARF(ALWAYS, "Total init latency: %fms", init_millis);

  if (micro_engine == NULL) {
    FARF(ALWAYS, "GetMicroEngineSingleton failed");
    return;
  }

  rpc::skel::FillRandomValue(input0, input_length * sizeof(float));
  micro_engine->RegisterInputData(0, input0, input0_shape);

  // warm up
  t0 = port::api::NowMicros();
  if (micro_engine->Run() != MACE_SUCCESS) {
    FARF(ALWAYS, "warm up error");
    return;
  } else {
    t1 = port::api::NowMicros();
    double run_millis = (t1 - t0) / 1000.0;
    FARF(ALWAYS, "run latency for cold start: %fms", run_millis);
  }

  // run
  t0 = port::api::NowMicros();
  for (int32_t i = 0; i < kMicroRunTestTimes; ++i) {
    micro_engine->Run();
  }
  t1 = port::api::NowMicros();

  double run_millis = (t1 - t0) / kMicroRunTestTimes / 1000.0;
  FARF(ALWAYS, "run latency: %fms", run_millis);
}

}  // namespace testing
}  // namespace micro

void MaceMcRun() {
  micro::testing::MicroRunModel();
}

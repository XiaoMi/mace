// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifdef MACE_ENABLE_OPENCL

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace test {

TEST(MaceAPIExceptionTest, WrongInputTest) {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  input_names.push_back(MakeString("input", 0));
  output_names.push_back(MakeString("output", 0));

  MaceEngineConfig config(DeviceType::GPU);
  config.SetGPUContext(
      ops::test::OpTestContext::Get()->gpu_context());

  std::shared_ptr<NetDef> net_def(new NetDef());
  for (size_t i = 0; i < input_names.size(); ++i) {
    InputOutputInfo *info = net_def->add_input_info();
    info->set_name(input_names[i]);
  }

  MaceEngine engine(config);
  ASSERT_DEATH(engine.Init(net_def.get(), {"input"}, output_names, nullptr),
               "");
}

}  // namespace test
}  // namespace mace

#endif  // MACE_ENABLE_OPENCL

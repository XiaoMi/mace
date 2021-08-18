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
#include <fstream>
#include <vector>
#include <iostream>

#include "data/data.h"
#include "micro.h"
namespace micro {
namespace MICRO_MODEL_NAME {

MaceStatus CreateMaceMicroEngineFromBinary(uint8_t *model_data,
                                           int32_t size,
                                           int32_t input_size,
                                           MaceMicroEngine **engine);

}
}  // namespace micro

int main(int args, char *argv[]) {
  if (args != 2) {
    printf("Please input model file");
    return -1;
  }

  std::ifstream model_file(argv[1], std::ios::binary);
  if (!model_file.is_open()) {
    printf("Failed to open file");
    return -1;
  }
  model_file.seekg(0, model_file.end);
  auto length = model_file.tellg();
  std::vector<uint8_t> model_bytes(length);
  model_file.seekg(0, model_file.beg);
  model_file.read(reinterpret_cast<char *>(model_bytes.data()), length);
  model_file.close();

  micro::MaceMicroEngine *micro_engine = nullptr;
  if (micro::MICRO_MODEL_NAME::CreateMaceMicroEngineFromBinary(
          model_bytes.data(), model_bytes.size(), 1, &micro_engine) !=
      micro::MACE_SUCCESS) {
    std::cout << "Failed to create engine" << std::endl;
    return -1;
  }

  if (micro_engine->RegisterInputData(0, MICRO_DATA_NAME::input,
                                      MICRO_DATA_NAME::input_dims) != micro::MACE_SUCCESS) {
    std::cout << "Failed to input" << std::endl;
    return -1;
  }

  if (micro_engine->Run() != micro::MACE_SUCCESS) {
    std::cout << "Failed to run" << std::endl;
    return -1;
  }

  float *output_buffer = NULL;
  const int32_t *output_dims = NULL;
  uint32_t dim_size = 0;
  micro_engine->GetOutputData(0, reinterpret_cast<void **>(&output_buffer),
                              &output_dims, &dim_size);

  int32_t output_total_size = 1;
  for (uint32_t i = 0; i < dim_size; ++i) {
    output_total_size *= output_dims[i];
  }

  for (int32_t i = 0; i < output_total_size; ++i) {
    printf("%d: %f\n", i, output_buffer[i]);
  }

  micro::DestroyMicroEngineFromBinary(micro_engine);

  return 0;
}

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

#include "data/data.h"
#include "micro.h"
#include "micro/framework/graph.h"
#include "micro/include/public/micro.h"
#include "micro/model/net_def.h"

#define IDENT(x) x
#define XSTR(x) #x
#define STR(x) XSTR(x)
#define PATH(x, y) STR(IDENT(x) IDENT(y))
#define MICRO_OPS_LIST_NAME /micro_ops_list.h

#include PATH(MICRO_MODEL_NAME, MICRO_OPS_LIST_NAME)

micro::MaceMicroEngine *createMaceMicroEngineFromBinary(
    uint8_t *model_data,
    int32_t size,
    micro::framework::Operator **op_array,
    int32_t input_num) {
  struct model_header {
    int64_t net_def_offset;
    int64_t graph_offset;
    int64_t model_offset;
    int64_t model_end;
    int64_t tensor_mem_size;
    int64_t scratch_buffer_size;
  };

  auto header = reinterpret_cast<model_header *>(model_data);
  auto net_def = reinterpret_cast<micro::model::NetDef *>(
      model_data + header->net_def_offset);
  auto model = reinterpret_cast<uint8_t *>(model_data + header->model_offset);
  auto graph = reinterpret_cast<micro::framework::Graph *>(
      model_data + header->graph_offset);

  auto tensor_mem = new uint8_t[header->tensor_mem_size];
  auto scratch_buffer = new uint8_t[header->scratch_buffer_size];

  // Supports only 1 input
  const void **input_buffers = new const void *[input_num];
  const int32_t **input_shapes = new const int32_t *[input_num];

  micro::MaceMicroEngine *micro_engine = new micro::MaceMicroEngine();
  micro::MaceMicroEngineConfig *engine_config =
      new micro::MaceMicroEngineConfig{net_def,
                                       model,
                                       graph,
                                       op_array,
                                       tensor_mem,
                                       input_buffers,
                                       input_shapes,
                                       scratch_buffer,
                                       header->scratch_buffer_size};
  micro_engine->Init(engine_config);

  return micro_engine;
}

void destroyMicroEngineFromBinary(micro::MaceMicroEngine *engine) {
  delete[] engine->GetEngineConfig()->tensor_mem_;
  delete[] engine->GetEngineConfig()->scratch_buffer_;
  delete[] engine->GetEngineConfig()->input_buffers_;
  delete[] engine->GetEngineConfig()->input_shapes_;
  delete engine->GetEngineConfig();
  delete engine;
}

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

  auto micro_engine =
      createMaceMicroEngineFromBinary(model_bytes.data(), model_bytes.size(),
                                      micro::MICRO_MODEL_NAME::kOpsArray, 1);

  micro_engine->RegisterInputData(0, MICRO_DATA_NAME::input,
                                  MICRO_DATA_NAME::input_dims);
  micro_engine->Run();

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

  destroyMicroEngineFromBinary(micro_engine);

  return 0;
}

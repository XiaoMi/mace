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

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "micro/base/logging.h"
#include "micro/framework/graph.h"
#include "micro/include/utils/macros.h"

#ifndef MICRO_MODEL_NAME
#error Please specify model name in the command
#endif

namespace micro {
namespace MICRO_MODEL_NAME {
extern uint8_t kGraphData[];
}  // namespace MICRO_MODEL_NAME

namespace framework {

#ifdef MACE_WRITE_MAGIC
#define MACE_CHECK_MAGIC_CODE(OBJ_NAME)                    \
  MACE_ASSERT1(CheckMagic(OBJ_NAME, OBJ_NAME->GetMagic(),  \
      OBJ_NAME->GetHardCodeMagic()), "CheckMagic failed.")

bool CheckMagic(const Serialize *serial_obj,
                SerialUint32 magic, SerialUint32 hard_code_magic) {
  char str_magic[5] = {0};
  serial_obj->MagicToString(magic, str_magic);
  bool succ = (magic == hard_code_magic);
  if (!succ) {
    char str_hc_magic[5] = {0};
    serial_obj->MagicToString(hard_code_magic, str_hc_magic);
    LOG(INFO) << "The magic is invalid, " << "magic = " << str_magic
              << ", hard_code_magic = " << str_hc_magic;
  } else {
    LOG(INFO) << "OK, The magic is " << str_magic;
  }
  return succ;
}
#else
#define MACE_CHECK_MAGIC_CODE(OBJ_NAME)
#endif

class GraphTest : public ::testing::Test {
};

void OutputOpContextInfo(const Graph *graph, const OpContext *op_context) {
  LOG(INFO) << "op_idx is: " << op_context->op_idx();
  uint32_t input_info_size = op_context->input_info_size();
  LOG(INFO) << "input_info size size is: " << input_info_size;
  for (uint32_t i = 0; i < input_info_size; ++i) {
    const OpIOInfo *input_info = op_context->input_info(i);
    graph->Uint2OpIOInfo(input_info);
    LOG(INFO) << "op_def_idx_: " << input_info->op_def_idx_
              << ", output_idx_: " << input_info->output_idx_;
  }
}

void OutputGraphInfo(const Graph *graph) {
  MACE_CHECK_MAGIC_CODE(graph);
  uint32_t op_context_size = graph->op_context_size();
  LOG(INFO) << "op_context size is: " << op_context_size;
  for (uint32_t i = 0; i < op_context_size; ++i) {
    OutputOpContextInfo(graph, graph->op_context(i));
  }

  uint32_t input_op_idx_size = graph->input_op_idx_size();
  LOG(INFO) << "input_op_idx size is: " << input_op_idx_size;
  for (uint32_t i = 0; i < input_op_idx_size; ++i) {
    LOG(INFO) << "input_op_idx=" << graph->input_op_idx(i);
  }

  uint32_t output_info_size = graph->output_info_size();
  LOG(INFO) << "output_info size is: " << output_info_size;
  for (uint32_t i = 0; i < output_info_size; ++i) {
    const OpIOInfo *output_info = graph->output_info(i);
    graph->Uint2OpIOInfo(output_info);
    LOG(INFO) << "op_def_idx_ is: " << output_info->op_def_idx_
              << ", output_idx_ is: " << output_info->output_idx_;
  }
}

void OutputAllInfo(const uint8_t *address) {
  const Graph *graph = reinterpret_cast<const Graph *>(address);
  MACE_ASSERT1(graph != NULL, "reinterpret_cast failed.");

  OutputGraphInfo(graph);
}


TEST_F(GraphTest, OutputAllInfo) {
  LOG(INFO) << "GraphTest start";
  OutputAllInfo(MICRO_MODEL_NAME::kGraphData);
}

}  // namespace framework
}  // namespace micro

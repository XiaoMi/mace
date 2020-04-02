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
#include "micro/include/utils/macros.h"
#include "micro/model/const_tensor.h"
#include "micro/model/net_def.h"
#include "micro/model/operator_def.h"

#ifndef MICRO_MODEL_NAME
#error Please specify model name in the command
#endif

namespace micro {
namespace MICRO_MODEL_NAME {
extern uint8_t kNetDef[];
}  // namespace MICRO_MODEL_NAME

namespace model {

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
#define MACE_CHECK_MAGIC_CODE(OBJ_NAME) MACE_UNUSED(OBJ_NAME)
#endif

class NetDefTest : public ::testing::Test {
};

void OutputArgumentInfo(const Argument *argument) {
  MACE_CHECK_MAGIC_CODE(argument);
  LOG(INFO) << "The argument name: " << argument->name();
}

void OutputOperatorInfo(const OperatorDef *op_def) {
  MACE_CHECK_MAGIC_CODE(op_def);
  LOG(INFO) << "The op_def name: " << op_def->name();
  uint32_t input_size = op_def->input_size();
  LOG(INFO) << "\tThe op_def input size: " << input_size;
  for (uint32_t j = 0; j < input_size; ++j) {
    LOG(INFO) << "\t\tThe input name: " << op_def->input(j);
  }
  auto output_size = op_def->output_size();
  LOG(INFO) << "\tThe op_def output size: " << output_size;
  for (uint32_t k = 0; k < output_size; ++k) {
    LOG(INFO) << "\t\tThe output name: " << op_def->output(k);
  }
  auto mem_offset_size = op_def->mem_offset_size();
  LOG(INFO) << "\tThe mem_offset size: " << mem_offset_size;
  for (uint32_t k = 0; k < mem_offset_size; ++k) {
    LOG(INFO) << "\t\tThe " << k << "th mem_offset: " << op_def->mem_offset(k);
  }
  auto arg_size = op_def->arg_size();
  LOG(INFO) << "\tThe arg size: " << arg_size;
  for (uint32_t k = 0; k < arg_size; ++k) {
    OutputArgumentInfo(op_def->arg(k));
  }
}

void OutputTensorInfo(const ConstTensor *tensor) {
  MACE_CHECK_MAGIC_CODE(tensor);
  LOG(INFO) << "The tensor name: " << tensor->name();

  auto dim_size = tensor->dim_size();
  LOG(INFO) << "\tThe tensor dim size: " << dim_size;
  for (uint32_t i = 0; i < dim_size; ++i) {
    LOG(INFO) << "\t\ttensor dim[" << i << "] = " << tensor->dim(i);
  }

  auto float_data_size = tensor->float_data_size();
  LOG(INFO) << "\tThe tensor float_data size: " << float_data_size;
  for (uint32_t i = 0; i < float_data_size; ++i) {
    const float f_value = tensor->float_data(i);
    LOG(INFO) << "\t\ttensor float_data[" << i << "] = " << f_value;
  }
  if (float_data_size > 0) {
    MACE_ASSERT(false);
  }
}

void OutputNetDefInfo(const NetDef *net_def) {
  MACE_CHECK_MAGIC_CODE(net_def);
  auto op_size = net_def->op_size();
  LOG(INFO) << "op size is: " << op_size;
  for (uint32_t i = 0; i < op_size; ++i) {
    OutputOperatorInfo(net_def->op(i));
  }

  auto arg_size = net_def->arg_size();
  LOG(INFO) << "arg size is: " << arg_size;
  auto arg_byte_size = sizeof(Argument);
  LOG(INFO) << "arg byte size is: " << (int32_t) arg_byte_size;
  for (uint32_t i = 0; i < arg_size; ++i) {
    OutputArgumentInfo(net_def->arg(i));
  }

  auto tensor_size = net_def->tensor_size();
  LOG(INFO) << "tensor size is: " << tensor_size;
  for (uint32_t i = 0; i < tensor_size; ++i) {
    OutputTensorInfo(net_def->tensor(i));
  }

  auto data_type = net_def->data_type();
  LOG(INFO) << "data_type is: " << data_type;

  auto input_info_size = net_def->input_info_size();
  LOG(INFO) << "input_info size is: " << input_info_size;
  for (uint32_t i = 0; i < input_info_size; ++i) {
    MACE_CHECK_MAGIC_CODE(net_def->input_info(i));
  }

  auto output_info_size = net_def->output_info_size();
  LOG(INFO) << "output_info size is: " << output_info_size;
  for (uint32_t i = 0; i < output_info_size; ++i) {
    MACE_CHECK_MAGIC_CODE(net_def->output_info(i));
  }
}

void OutputAllInfo(const uint8_t *address) {
  const NetDef *net_def = reinterpret_cast<const NetDef *>(address);
  MACE_ASSERT1(net_def != NULL, "reinterpret_cast failed.");

  OutputNetDefInfo(net_def);
}

TEST_F(NetDefTest, OutputAllInfo) {
  LOG(INFO) << "NetDefTest start";
  OutputAllInfo(MICRO_MODEL_NAME::kNetDef);
}

}  // namespace model
}  // namespace micro

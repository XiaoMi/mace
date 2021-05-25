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

#include "mace/core/ops/operator.h"

#include <vector>

#include "mace/core/ops/op_construct_context.h"
#include "mace/core/ops/op_init_context.h"

namespace mace {
Operation::Operation(OpConstructContext *context)
    : operator_def_(context->operator_def()) {}

MaceStatus Operation::Init(OpInitContext *context) {
  Workspace *ws = context->workspace();
  for (const std::string &input_str : operator_def_->input()) {
    const Tensor *tensor = ws->GetTensor(input_str);
    MACE_CHECK(tensor != nullptr, "op ", operator_def_->type(),
               ": Encountered a non-existing input tensor: ", input_str);
    inputs_.push_back(tensor);
  }
  auto runtime = context->runtime();
  auto cur_mem_type = runtime->GetUsedMemoryType();
  for (int i = 0; i < operator_def_->output_size(); ++i) {
    const std::string output_str = operator_def_->output(i);
    if (ws->HasTensor(output_str)) {
      outputs_.push_back(ws->GetTensor(output_str));
    } else {
      MACE_CHECK(
          operator_def_->output_type_size() == 0 ||
              operator_def_->output_size() == operator_def_->output_type_size(),
          "operator output size != operator output type size",
          operator_def_->output_size(),
          operator_def_->output_type_size());
      DataType output_type;
      if (i < operator_def_->output_type_size()) {
        output_type = operator_def_->output_type(i);
      } else {
        output_type = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *operator_def_, "T", static_cast<int>(DT_FLOAT)));
      }

      const auto mem_type = static_cast<MemoryType>(
          ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
              *operator_def_, OutputMemoryTypeTagName(),
              static_cast<int>(cur_mem_type)));

      auto tensor_runtime = context->GetRuntimeByMemType(mem_type);
      outputs_.push_back(MACE_CHECK_NOTNULL(ws->CreateTensor(
          output_str, tensor_runtime, output_type, false, mem_type)));
    }
    if (i < operator_def_->output_shape_size()) {
      std::vector<index_t>
          shape_configured(operator_def_->output_shape(i).dims_size());
      for (size_t dim = 0; dim < shape_configured.size(); ++dim) {
        shape_configured[dim] = operator_def_->output_shape(i).dims(dim);
      }
      ws->GetTensor(output_str)->SetShapeConfigured(shape_configured);
    }
  }

  for (int i = 0; i < operator_def_->input_size(); ++i) {
    const std::string input_str = operator_def_->input(i);
    if (ws->HasTensor(input_str)) {
      Tensor *input = ws->GetTensor(input_str);
      const auto content_type = GetInputTensorContentType(i);
      input->SetContentType(content_type);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Operation::Forward(OpContext *context) {
  context->runtime()->ReleaseAllBuffer(RENT_SCRATCH);
  if (runtime_type() != RuntimeType::RT_CPU) {
    return Run(context);
  }

  for (auto iter = inputs_.begin(); iter != inputs_.end(); ++iter) {
    const Tensor *input = *iter;
    if (input->memory_type() != MemoryType::CPU_BUFFER) {
      input->Map(true);
    }
  }

  for (auto iter = outputs_.begin(); iter != outputs_.end(); ++iter) {
    Tensor *output = *iter;
    if (output->memory_type() != MemoryType::CPU_BUFFER) {
      output->Map(true);
    }
  }

  auto ret = Run(context);

  for (auto iter = outputs_.begin(); iter != outputs_.end(); ++iter) {
    Tensor *output = *iter;
    if (output->memory_type() != MemoryType::CPU_BUFFER) {
      output->UnMap();
    }
  }

  for (auto iter = inputs_.begin(); iter != inputs_.end(); ++iter) {
    const Tensor *input = *iter;
    if (input->memory_type() != MemoryType::CPU_BUFFER) {
      input->UnMap();
    }
  }

  return ret;
}

int Operation::ReuseTensorMapId(size_t output_idx) const {
  MACE_UNUSED(output_idx);
  return -1;
}

BufferContentType Operation::GetInputTensorContentType(size_t idx) const {
  MACE_UNUSED(idx);
  return BufferContentType::IN_OUT_CHANNEL;
}

}  // namespace mace

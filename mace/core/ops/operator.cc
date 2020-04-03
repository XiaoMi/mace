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
      outputs_.push_back(MACE_CHECK_NOTNULL(ws->CreateTensor(
          output_str, context->device()->allocator(), output_type)));
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
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

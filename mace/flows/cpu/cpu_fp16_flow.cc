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


#include "mace/flows/cpu/cpu_fp16_flow.h"

#include "mace/core/flow/flow_registry.h"
#include "mace/core/fp16.h"
#include "mace/utils/transpose.h"

namespace mace {

CpuFp16Flow::CpuFp16Flow(FlowContext *flow_context)
    : CpuRefFlow(flow_context) {}

MaceStatus CpuFp16Flow::TransposeInputByDims(const MaceTensor &mace_tensor,
                                             Tensor *input_tensor,
                                             const std::vector<int> &dst_dims) {
  DataType input_dt = input_tensor->dtype();
  bool transposed = false;
  if (!dst_dims.empty()) {
    if (input_dt == DataType::DT_FLOAT16) {
      auto user_dt = mace_tensor.data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard input_guard(input_tensor);
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, mace_tensor.data<float>().get(), mace_tensor.shape(),
            dst_dims, input_tensor->mutable_data<half>()));
        transposed = true;
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard input_guard(input_tensor);
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, mace_tensor.data<half>().get(), mace_tensor.shape(),
            dst_dims, input_tensor->mutable_data<half>()));
        transposed = true;
      }
    }
  } else {
    if (input_dt == DataType::DT_FLOAT16) {
      auto user_dt = mace_tensor.data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard input_guard(input_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, mace_tensor.data<float>().get(),
            input_tensor->mutable_data<half>(), input_tensor->size());
        transposed = true;
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard input_guard(input_tensor);
        ops::CopyDataBetweenSameType(
            thread_pool_, mace_tensor.data<void>().get(),
            input_tensor->mutable_data<void>(), input_tensor->raw_size());
        transposed = true;
      }
    }
  }

  if (!transposed) {
    return CommonFp32Flow::TransposeInputByDims(mace_tensor, input_tensor,
                                                dst_dims);
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus CpuFp16Flow::TransposeOutputByDims(
    const mace::Tensor &output_tensor,
    MaceTensor *mace_tensor, const std::vector<int> &dst_dims) {
  bool transposed = false;
  auto output_dt = output_tensor.dtype();
  if (!dst_dims.empty()) {
    if (output_dt == DataType::DT_FLOAT16) {
      auto user_dt = mace_tensor->data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard output_guard(&output_tensor);
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, output_tensor.data<half>(), output_tensor.shape(),
            dst_dims, mace_tensor->data<float>().get()));
        transposed = true;
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, output_tensor.data<half>(), output_tensor.shape(),
            dst_dims, mace_tensor->data<half>().get()));
        transposed = true;
      }
    }
  } else {
    if (output_dt == DataType::DT_FLOAT16) {
      auto user_dt = mace_tensor->data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard output_guard(&output_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, output_tensor.data<half>(),
            mace_tensor->data<float>().get(), output_tensor.size());
        transposed = true;
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        ops::CopyDataBetweenSameType(
            thread_pool_, output_tensor.data<void>(),
            mace_tensor->data<void>().get(), output_tensor.raw_size());
        transposed = true;
      }
    }
  }

  if (!transposed) {
    return CommonFp32Flow::TransposeOutputByDims(output_tensor, mace_tensor,
                                                 dst_dims);
  }
  return MaceStatus::MACE_SUCCESS;
}

void RegisterCpuFp16Flow(FlowRegistry *flow_registry) {
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_CPU,
                     FlowSubType::FW_SUB_FP16, CpuFp16Flow);
}

}  // namespace mace

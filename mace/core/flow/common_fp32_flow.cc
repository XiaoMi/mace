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


#include "mace/core/flow/common_fp32_flow.h"

#include "mace/core/bfloat16.h"
#include "mace/core/types.h"
#include "mace/utils/transpose.h"

namespace mace {

CommonFp32Flow::CommonFp32Flow(FlowContext *flow_context)
    : BaseFlow(flow_context) {
  VLOG(3) << "CommonFp32Flow::CommonFp32Flow";
}

MaceStatus CommonFp32Flow::TransposeInputByDims(
    const MaceTensor &mace_tensor,
    Tensor *input_tensor, const std::vector<int> &dst_dims) {
  DataType input_dt = input_tensor->dtype();
  bool transposed = false;
  if (!dst_dims.empty()) {
    if (input_dt == DataType::DT_FLOAT) {
      MACE_CHECK(net_data_type_ == DT_FLOAT || net_data_type_ == DT_HALF
                 || net_data_type_ == DT_BFLOAT16);
      auto user_dt = mace_tensor.data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard input_guard(input_tensor);
        auto input_data = input_tensor->mutable_data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, mace_tensor.data<float>().get(),
            mace_tensor.shape(), dst_dims, input_data));
        transposed = true;
#ifdef MACE_ENABLE_FP16
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard input_guard(input_tensor);
        auto input_data = input_tensor->mutable_data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, mace_tensor.data<half>().get(),
            mace_tensor.shape(), dst_dims, input_data));
        transposed = true;
#endif
#ifdef MACE_ENABLE_BFLOAT16
      } else if (user_dt == IDT_BFLOAT16) {
        Tensor::MappingGuard input_guard(input_tensor);
        auto input_data = input_tensor->mutable_data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, mace_tensor.data<BFloat16>().get(),
            mace_tensor.shape(), dst_dims, input_data));
        transposed = true;
#endif
      }
    }
  } else {
    if (input_dt == DataType::DT_FLOAT) {
      MACE_CHECK(net_data_type_ == DT_FLOAT || net_data_type_ == DT_HALF);
      auto user_dt = mace_tensor.data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard input_guard(input_tensor);
        ops::CopyDataBetweenSameType(
            thread_pool_, mace_tensor.data<float>().get(),
            input_tensor->mutable_data<float>(), input_tensor->raw_size());
        transposed = true;
      } else if (user_dt == IDT_FLOAT16) {
#ifdef MACE_ENABLE_FP16
        Tensor::MappingGuard input_guard(input_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, mace_tensor.data<half>().get(),
            input_tensor->mutable_data<float>(), input_tensor->size());
        transposed = true;
#endif
#ifdef MACE_ENABLE_BFLOAT16
      } else if (user_dt == IDT_BFLOAT16) {
        Tensor::MappingGuard input_guard(input_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, mace_tensor.data<BFloat16>().get(),
            input_tensor->mutable_data<float>(), input_tensor->size());
        transposed = true;
#endif
      }
    }
  }

  if (!transposed) {
    return BaseFlow::TransposeInputByDims(mace_tensor, input_tensor, dst_dims);
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus CommonFp32Flow::TransposeOutputByDims(
    const mace::Tensor &output_tensor,
    MaceTensor *mace_tensor, const std::vector<int> &dst_dims) {
  bool transposed = false;
  auto output_dt = output_tensor.dtype();
  if (!dst_dims.empty()) {
    if (output_dt == DataType::DT_FLOAT) {
      auto user_dt = mace_tensor->data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard output_guard(&output_tensor);
        auto output_data = output_tensor.data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, output_data, output_tensor.shape(),
            dst_dims, mace_tensor->data<float>().get()));
        transposed = true;
#ifdef MACE_ENABLE_FP16
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        auto output_data = output_tensor.data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, output_data, output_tensor.shape(),
            dst_dims, mace_tensor->data<half>().get()));
        transposed = true;
#endif
#ifdef MACE_ENABLE_BFLOAT16
      } else if (user_dt == IDT_BFLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        auto output_data = output_tensor.data<float>();
        MACE_RETURN_IF_ERROR(ops::Transpose(
            thread_pool_, output_data, output_tensor.shape(),
            dst_dims, mace_tensor->data<BFloat16>().get()));
        transposed = true;
#endif
      }
    }
  } else {
    if (output_dt == DataType::DT_FLOAT) {
      auto user_dt = mace_tensor->data_type();
      if (user_dt == IDT_FLOAT) {
        Tensor::MappingGuard output_guard(&output_tensor);
        ops::CopyDataBetweenSameType(
            thread_pool_, output_tensor.data<float>(),
            mace_tensor->data<float>().get(), output_tensor.raw_size());
        transposed = true;
#ifdef MACE_ENABLE_FP16
      } else if (user_dt == IDT_FLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, output_tensor.data<float>(),
            mace_tensor->data<half>().get(), output_tensor.size());
        transposed = true;
#endif
#ifdef MACE_ENABLE_BFLOAT16
      } else if (user_dt == IDT_BFLOAT16) {
        Tensor::MappingGuard output_guard(&output_tensor);
        ops::CopyDataBetweenDiffType(
            thread_pool_, output_tensor.data<float>(),
            mace_tensor->data<BFloat16>().get(), output_tensor.size());
        transposed = true;
#endif
      }
    }
  }

  if (!transposed) {
    return BaseFlow::TransposeOutputByDims(output_tensor,
                                           mace_tensor, dst_dims);
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

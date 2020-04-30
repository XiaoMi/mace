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

#include "mace/core/net/allocate_strategy.h"

#include "mace/core/tensor.h"
#include "mace/utils/logging.h"

namespace mace {

namespace {
struct MemBlock {
  int refs;

  explicit MemBlock(Tensor *tensor_ptr) : refs(1), tensor(tensor_ptr) {}
  void AllocateBuffer() {
    if (tensor->memory<void>() == nullptr) {
      auto *runtime = tensor->GetCurRuntime();
      runtime->AllocateBufferForTensor(tensor, RENT_SHARE);
    }
  }

  void DeleteBuffer() {
    MACE_CHECK(refs == 0);
    if (tensor->memory<void>() != nullptr) {
      auto *runtime = tensor->GetCurRuntime();
      runtime->ReleaseBufferForTensor(tensor, RENT_SHARE);
    }
  }

 private:
  Tensor *tensor;
};
}  // namespace


template <>
MaceStatus AllocateTensorMemory<SERIAL_REF>(const OperationArray &operators) {
  std::unordered_map<std::string, std::shared_ptr<MemBlock>> tensor_refs;
  // Collect the refs of input tensor
  for (auto &op : operators) {
    size_t input_size = static_cast<size_t>(op->InputSize());
    for (size_t i = 0; i < input_size; ++i) {
      const Tensor *tensor = op->Input(i);
      if (tensor->is_weight()) {
        continue;
      }
      auto tensor_name = tensor->name();
      if (tensor_refs.count(tensor_name) == 0) {
        tensor_refs.emplace(tensor_name, std::make_shared<MemBlock>(
            const_cast<Tensor *>(tensor)));
      } else {
        tensor_refs[tensor_name]->refs++;
      }
    }
  }

  // Merge the refs that reuse the buffer
  for (auto &op : operators) {
    size_t output_size = static_cast<size_t>(op->OutputSize());
    for (size_t i = 0; i < output_size; ++i) {
      Tensor *out_tensor = op->Output(i);
      auto out_tensor_name = out_tensor->name();
      int reuse_input_idx = op->ReuseTensorMapId(i);
      if (reuse_input_idx >= 0) {
        const Tensor *reuse_in_tensor = op->Input(reuse_input_idx);
        if (reuse_in_tensor->is_weight()) {
          continue;
        }
        if (tensor_refs.count(out_tensor_name) == 0) {
          VLOG(2) << "tensor " << out_tensor_name << " is model's output";
          continue;
        }

        // Merge the refs
        auto reuse_in_tensor_name = reuse_in_tensor->name();
        tensor_refs[reuse_in_tensor_name]->refs +=
            tensor_refs.at(out_tensor_name)->refs;
        tensor_refs[out_tensor_name] = tensor_refs.at(reuse_in_tensor_name);
      }
    }
  }

  // Simulate the execution of net and allocate memory for tensor
  for (auto &op : operators) {
    VLOG(2) << "Operator " << op->debug_def().name() << "<"
            << op->runtime_type() << ", " << op->debug_def().type() << ">";
    size_t output_size = static_cast<size_t>(op->OutputSize());
    for (size_t i = 0; i < output_size; ++i) {
      Tensor *tensor = op->Output(i);
      auto *runtime = tensor->GetCurRuntime();
      auto tensor_name = tensor->name();
      VLOG(2) << "allocate buffer for tensor: " << tensor_name << ", "
              << MakeString(tensor->shape()) << ", runtime_type: "
              << runtime->GetRuntimeType()
              << ", mem type: " << tensor->memory_type()
              << ", data format: " << static_cast<int>(tensor->data_format());
      if (tensor_refs.count(tensor_name) == 0) {
        tensor_refs.emplace(tensor_name, std::make_shared<MemBlock>(tensor));
        VLOG(2) << "tensor " << tensor_name << " is model's output";
      }
      tensor_refs.at(tensor_name)->AllocateBuffer();

      auto data_format = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          op->debug_def(), "data_format", static_cast<int>(DataFormat::NONE));
      tensor->set_data_format(static_cast<DataFormat>(data_format));
    }

    size_t input_size = static_cast<size_t>(op->InputSize());
    for (size_t i = 0; i < input_size; ++i) {
      const Tensor *tensor = op->Input(i);
      auto tensor_name = tensor->name();
      if (tensor->is_weight()) {
        continue;
      }
      MACE_CHECK(tensor_refs.count(tensor_name) > 0);
      int ref_num = tensor_refs.at(tensor_name)->refs;
      MACE_CHECK(ref_num > 0);
      tensor_refs[tensor_name]->refs = ref_num - 1;
      if (ref_num == 1) {
        auto runtime = tensor->GetCurRuntime();
        VLOG(2) << "net before release buffer for tensor: "
                << tensor->name() << ", ref = " << ref_num << ", "
                << tensor->data<void>() << ", runtime_type: "
                << runtime->GetRuntimeType();
        tensor_refs.at(tensor_name)->DeleteBuffer();
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

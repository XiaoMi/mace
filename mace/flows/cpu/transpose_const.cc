// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

#include "mace/core/workspace.h"
#include "mace/core/proto/arg_helper.h"
#include "mace/flows/cpu/transpose_const.h"
#include "mace/proto/mace.pb.h"
#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"

namespace mace {

namespace {
static std::vector<index_t> GetTensorStride(const Tensor *tensor) {
    int32_t ndim = static_cast<int32_t>(tensor->dim_size());
    std::vector<index_t> stride(ndim, 1);
    for (int32_t i = ndim - 2; i >= 0; --i) {
      stride[i] = stride[i+1] * static_cast<index_t>(tensor->dim(i+1));
    }
    return stride;
  }
}

MaceStatus DoTransposeConstForCPU(
    mace::utils::ThreadPool *thread_pool,
    Workspace *ws,
    Runtime *runtime,
    OperatorDef *op_def,
    const int input_idx) {

  MACE_UNUSED(thread_pool);
  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);

  MemoryType src_mem_type = input->memory_type();
  DataType src_dt = input->dtype();
  MACE_CHECK(src_mem_type == CPU_BUFFER || src_mem_type == GPU_BUFFER,
             "Only support transform CPU_BUFFER or GPU_BUFFER,",
             " but ", input_name, " has ", static_cast<int>(src_mem_type));
  DataType dst_dt = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *op_def, "T", static_cast<int>(DataType::DT_FLOAT)));
  MACE_CHECK((src_dt == DT_FLOAT || src_dt == DT_HALF) &&
             (dst_dt == DT_FLOAT || dst_dt == DT_HALF),
             "Only support transform DT_FLOAT or DT_HALF,",
             " but ", input_name, " has ", static_cast<int>(src_dt));
  // Only support: half/float -> float, half -> half.
  // Float -> half is not supported.
  if (dst_dt == DT_HALF) {
    MACE_CHECK(src_dt == DT_HALF, "When dst_dt is DT_HALF, "
               "src_dt must be DT_HALF, but ", input_name,
               " has data type of ", static_cast<int>(src_dt));
  }
  MemoryType dst_mem_type = CPU_BUFFER;
  std::vector<index_t> input_shape = input->shape();
  std::vector<index_t> output_shape = input_shape;
  DataFormat op_data_format =
      static_cast<DataFormat>(ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *op_def, "data_format",
          static_cast<int>(DataFormat::NONE)));
  bool cpu_nhwc = (op_data_format == DataFormat::NHWC &&
                   input_shape.size() == 4);
  bool cpu_nchw = (op_data_format == DataFormat::NCHW &&
                   input_shape.size() == 4);
  if (src_mem_type == dst_mem_type &&
      src_dt == dst_dt &&
      (input_shape.size() != 4 || cpu_nhwc)) {
    return MaceStatus::MACE_SUCCESS;
  }
  std::vector<int> dims = {0, 3, 1, 2};
  if (cpu_nchw) {
    for (size_t i = 0; i < dims.size(); ++i) {
      output_shape[i] = input_shape[dims[i]];
    }
  }
  std::string output_name = input_name + std::string("_const_used_by_cpu");
  Tensor *output = ws->GetTensor(output_name);
  MACE_CHECK(output == nullptr || output->shape() == output_shape,
             output_name, " should not exist, ",
             "or it must be of shape ", MakeString(output_shape));
  bool already_transposed = (output != nullptr &&
                             output->shape() == output_shape);
  if (output == nullptr) {
    std::unique_ptr<Tensor> output_tensor =
        make_unique<Tensor>(runtime, dst_dt, dst_mem_type,
                            output_shape, true, output_name);
    output = output_tensor.get();
    runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
    ws->AddTensor(output_name, std::move(output_tensor));
  }
  op_def->set_input(input_idx, output_name);
  if (already_transposed) {
    return MaceStatus::MACE_SUCCESS;
  }
  input->Map(true);
  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  size_t num_elem = input->size();
  size_t dst_bytes = output->raw_size();

  if (input_shape.size() != 4 || cpu_nhwc) {
    if (src_dt == dst_dt) {
      memcpy(reinterpret_cast<void*>(output_data),
             reinterpret_cast<const void*>(input_data), dst_bytes);
    } else if (src_dt == DT_HALF && dst_dt == DT_FLOAT) {  // half->float
      // Can only be cpu/gpu half to cpu float, no matter 4D or non-4D
      const half *half_input = reinterpret_cast<const half*>(input_data);
      thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = half_float::half_cast<float>(half_input[i]);
        }
      }, 0, num_elem, 1);
    }
    return MaceStatus::MACE_SUCCESS;
  }
  index_t N = input_shape[0];
  index_t H = input_shape[1];
  index_t W = input_shape[2];
  index_t C = input_shape[3];
  std::vector<index_t> input_stride = GetTensorStride(input);
  std::vector<index_t> output_stride = GetTensorStride(output);
  if (src_dt == DT_HALF && dst_dt == DT_FLOAT) {
    const half *input_ptr = reinterpret_cast<const half*>(input_data);
    float *output_ptr = reinterpret_cast<float*>(output_data);
    thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        index_t n = i / C;
        index_t c = i - n * C;
        index_t hw_base = n * output_stride[0] + c * output_stride[1];
        index_t in_idx_nc = n * input_stride[0] + c;

        for (index_t h = 0; h < H; ++h) {
          index_t w_base = hw_base + h * output_stride[2];
          index_t in_idx_nhc = in_idx_nc + h * input_stride[1];

          for (index_t w = 0; w < W; ++w) {
            index_t in_idx = in_idx_nhc + w * input_stride[2];
            index_t out_idx = w_base + w;

            output_ptr[out_idx] =
                half_float::half_cast<float>(input_ptr[in_idx]);
          }
        }
      }
    }, 0, N * C, 1);

  } else if (src_dt == dst_dt) {
    // Memcpy can deal with float -> float and half -> half.
    const char *input_ptr = reinterpret_cast<const char*>(input_data);
    char *output_ptr = reinterpret_cast<char*>(output_data);
    int elem_size = input->raw_size() / input->size();
    thread_pool->Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        index_t n = i / C;
        index_t c = i - n * C;
        index_t hw_base = n * output_stride[0] + c * output_stride[1];
        index_t in_idx_nc = n * input_stride[0] + c;

        for (index_t h = 0; h < H; ++h) {
          index_t w_base = hw_base + h * output_stride[2];
          index_t in_idx_nhc = in_idx_nc + h * input_stride[1];

          for (index_t w = 0; w < W; ++w) {
            index_t in_idx = in_idx_nhc + w * input_stride[2];
            index_t out_idx = w_base + w;

            memcpy(output_ptr + elem_size * out_idx,
                   input_ptr + elem_size * in_idx, elem_size);
          }
        }
      }
    }, 0, N * C, 1);
  } else {
    LOG(FATAL) << "Transposing from DT_FLOAT to DT_HALF is not supported";
  }
  input->MarkUnused();
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TransposeConstForCPU(
    mace::utils::ThreadPool *thread_pool,
    Workspace *ws,
    Runtime *runtime,
    NetDef *net_def) {
  // Must be same types in transformer.py,
  // and more types may be added in the future.
  std::unordered_set<std::string> equal_types = {"Eltwise", "Concat"};
  int num_ops = net_def->op_size();

  for (int idx = 0; idx < num_ops; ++idx) {
    OperatorDef *op_def = net_def->mutable_op(idx);
    int end_idx = 1;
    if (equal_types.find(op_def->type()) != equal_types.end()) {
      end_idx =  op_def->input_size();
    }
    RuntimeType op_device_type = static_cast<RuntimeType>(
        op_def->device_type());
    if (RuntimeType::RT_CPU != op_device_type) {
      continue;
    }
    for (int input_idx = 0; input_idx < end_idx; ++input_idx) {
      Tensor *tensor = ws->GetTensor(op_def->input(input_idx));
      if (!(tensor != nullptr && tensor->is_weight())) {
        continue;
      }
      MACE_RETURN_IF_ERROR(DoTransposeConstForCPU(thread_pool, ws, runtime,
                                                 op_def, input_idx));
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace

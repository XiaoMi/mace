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


#include "mace/flows/opencl/opencl_ref_flow.h"

#include <memory>

#include "mace/core/flow/flow_registry.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/opencl_runtime.h"
#include "mace/runtimes/opencl/transform/buffer_transformer.h"

namespace mace {

namespace {
std::unique_ptr<Tensor> CreateOpenclTensor(const MaceTensor &mace_tensor,
                                           const Tensor *link_tensor) {
  auto mem_type = mace_tensor.memory_type();
  MACE_CHECK(mem_type == GPU_IMAGE || mem_type == GPU_BUFFER);
  auto data_type = static_cast<DataType>(mace_tensor.data_type());
  auto &shape = mace_tensor.shape();
  auto buffer = make_unique<Buffer>(mem_type, data_type, shape,
                                    mace_tensor.data<void>().get());
  auto runtime = link_tensor->GetCurRuntime();
  auto tensor = make_unique<Tensor>(runtime, data_type, mem_type, shape);
  runtime->SetBufferToTensor(std::move(buffer), tensor.get());

  return tensor;
}

}  // namespace

OpenclRefFlow::OpenclRefFlow(FlowContext *flow_context)
    : CpuRefFlow(flow_context) {
  VLOG(3) << "OpenclRefFlow::OpenclRefFlow";
}

MaceStatus OpenclRefFlow::TransposeInputByDims(
    const MaceTensor &mace_tensor,
    Tensor *input_tensor, const std::vector<int> &dst_dims) {
  MaceStatus status = MaceStatus::MACE_UNSUPPORTED;
  MemoryType mem_type = mace_tensor.memory_type();
  MACE_CHECK(mem_type != GPU_IMAGE, "Haven't support GPU_IMAGE now.");
  if (mem_type == GPU_BUFFER) {
    DataType input_dt = input_tensor->dtype();
    MACE_CHECK(input_dt == DT_FLOAT || input_dt == DT_HALF);
    MACE_CHECK(mace_tensor.data_format() == DataFormat::NHWC,
               "Only support NHWC for GPU input.");
    auto user_tensor = CreateOpenclTensor(mace_tensor, input_tensor);
    auto transformer = OpenCLBufferTransformer(mem_type,
                                               input_tensor->memory_type());
    OpContext op_context(ws_.get(), input_tensor->GetCurRuntime());
    // TODO(luxuhui): op_context.set_future and stat performance
    status = transformer.Transform(
        &op_context, user_tensor.get(), IN_OUT_CHANNEL,
        input_tensor->memory_type(), 0, input_tensor);
  } else {  // mem_type == MemoryType::CPU_BUFFER
    status = CommonFp32Flow::TransposeInputByDims(mace_tensor, input_tensor,
                                                  dst_dims);
  }

  return status;
}

MaceStatus OpenclRefFlow::TransposeOutputByDims(
    const mace::Tensor &output_tensor,
    MaceTensor *mace_tensor, const std::vector<int> &dst_dims) {
  MaceStatus status = MaceStatus::MACE_UNSUPPORTED;
  MemoryType mem_type = mace_tensor->memory_type();
  MACE_CHECK(mem_type != GPU_IMAGE, "Haven't support GPU_IMAGE now.");
  if (mem_type == GPU_BUFFER) {
    DataType output_dt = output_tensor.dtype();
    MACE_CHECK(output_dt == DT_FLOAT || output_dt == DT_HALF);
    MACE_CHECK(mace_tensor->data_format() == DataFormat::NHWC,
               "Only support NHWC for GPU output.");
    auto user_tensor = CreateOpenclTensor(*mace_tensor, &output_tensor);
    auto transformer = OpenCLBufferTransformer(output_tensor.memory_type(),
                                               mem_type);
    OpContext op_context(ws_.get(), output_tensor.GetCurRuntime());
    // TODO(luxuhui): op_context.set_future and stat performance
    status = transformer.Transform(&op_context, &output_tensor, IN_OUT_CHANNEL,
                                   mem_type, 0, user_tensor.get());
  } else {
    status = CommonFp32Flow::TransposeOutputByDims(output_tensor,
                                                   mace_tensor, dst_dims);
  }

  return status;
}

MaceStatus OpenclRefFlow::GetInputTransposeDims(
    const std::pair<const std::string, MaceTensor> &input,
    const Tensor *input_tensor,
    std::vector<int> *dst_dims, DataFormat *data_format) {
  *data_format = input.second.data_format();
  if (input_tensor->data_format() != DataFormat::NONE) {
    if (input.second.shape().size() == 4 &&
        input.second.data_format() == DataFormat::NCHW) {
      VLOG(1) << "Transform input " << input.first << " from NCHW to NHWC";
      *data_format = DataFormat::NHWC;
      *dst_dims = {0, 2, 3, 1};
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

void OpenclRefFlow::AfterRun() {
  auto *opencl_runtime = static_cast<OpenclRuntime *>(main_runtime_);
  auto *opencl_executor = opencl_runtime->GetOpenclExecutor();
  opencl_executor->command_queue().finish();
  opencl_executor->SaveBuiltCLProgram();
}
MaceStatus OpenclRefFlow::Run(TensorMap *input_tensors,
                              TensorMap *output_tensors,
                              RunMetadata *run_metadata) {
  MACE_RETURN_IF_ERROR(CpuRefFlow::Run(
      input_tensors, output_tensors, run_metadata));
  AfterRun();
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclRefFlow::FakeWarmup() {
  MACE_RETURN_IF_ERROR(net_->Run(nullptr, true));
  AfterRun();
  return MaceStatus::MACE_SUCCESS;
}

void RegisterOpenclRefFlow(FlowRegistry *flow_registry) {
  MACE_REGISTER_FLOW(flow_registry, RuntimeType::RT_OPENCL,
                     FlowSubType::FW_SUB_REF, OpenclRefFlow);
}

}  // namespace mace

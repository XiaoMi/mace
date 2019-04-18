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

#include "mace/ops/ops_test_util.h"
#include "mace/core/memory_optimizer.h"
#include "mace/utils/memory.h"
#include "mace/core/net_def_adapter.h"

namespace mace {
namespace ops {
namespace test {

OpDefBuilder::OpDefBuilder(const char *type, const std::string &name) {
  op_def_.set_type(type);
  op_def_.set_name(name);
}

OpDefBuilder &OpDefBuilder::Input(const std::string &input_name) {
  op_def_.add_input(input_name);
  return *this;
}

OpDefBuilder &OpDefBuilder::Output(const std::string &output_name) {
  op_def_.add_output(output_name);
  return *this;
}

OpDefBuilder &OpDefBuilder::OutputType(
    const std::vector<DataType> &output_type) {
  for (auto out_t : output_type) {
    op_def_.add_output_type(out_t);
  }
  return *this;
}

OpDefBuilder &OpDefBuilder::OutputShape(
    const std::vector<mace::index_t> &output_shape) {
  auto shape = op_def_.add_output_shape();
  for (auto s : output_shape) {
    shape->add_dims(s);
  }
  return *this;
}

OpDefBuilder OpDefBuilder::AddIntArg(const std::string &name, const int value) {
  auto arg = op_def_.add_arg();
  arg->set_name(name);
  arg->set_i(value);
  return *this;
}

OpDefBuilder OpDefBuilder::AddFloatArg(const std::string &name,
                                       const float value) {
  auto arg = op_def_.add_arg();
  arg->set_name(name);
  arg->set_f(value);
  return *this;
}

OpDefBuilder OpDefBuilder::AddStringArg(const std::string &name,
                                        const char *value) {
  auto arg = op_def_.add_arg();
  arg->set_name(name);
  arg->set_s(value);
  return *this;
}

OpDefBuilder OpDefBuilder::AddIntsArg(const std::string &name,
                                      const std::vector<int> &values) {
  auto arg = op_def_.add_arg();
  arg->set_name(name);
  for (auto value : values) {
    arg->add_ints(value);
  }
  return *this;
}

OpDefBuilder OpDefBuilder::AddFloatsArg(const std::string &name,
                                        const std::vector<float> &values) {
  auto arg = op_def_.add_arg();
  arg->set_name(name);
  for (auto value : values) {
    arg->add_floats(value);
  }
  return *this;
}

void OpDefBuilder::Finalize(OperatorDef *op_def) const {
  MACE_CHECK(op_def != nullptr, "input should not be null.");
  *op_def = op_def_;
}

namespace {
#ifdef MACE_ENABLE_OPENCL
std::string GetStoragePathFromEnv() {
  char *storage_path_str = getenv("MACE_INTERNAL_STORAGE_PATH");
  if (storage_path_str == nullptr) return "";
  return storage_path_str;
}
#endif
}  // namespace

OpTestContext *OpTestContext::Get(int num_threads,
                                  CPUAffinityPolicy cpu_affinity_policy) {
  static OpTestContext instance(num_threads,
                                cpu_affinity_policy);
  return &instance;
}

OpTestContext::OpTestContext(int num_threads,
                             CPUAffinityPolicy cpu_affinity_policy)
#ifdef MACE_ENABLE_OPENCL
    : gpu_context_(std::make_shared<GPUContext>(GetStoragePathFromEnv())),
      opencl_mem_types_({MemoryType::GPU_IMAGE}),
      thread_pool_(make_unique<utils::ThreadPool>(num_threads,
                                                  cpu_affinity_policy)) {
#else
    : thread_pool_(make_unique<utils::ThreadPool>(num_threads,
                                                  cpu_affinity_policy)) {
#endif
  thread_pool_->Init();

  device_map_[DeviceType::CPU] = make_unique<CPUDevice>(
      num_threads, cpu_affinity_policy, thread_pool_.get());

#ifdef MACE_ENABLE_OPENCL
  device_map_[DeviceType::GPU] = make_unique<GPUDevice>(
      gpu_context_->opencl_tuner(),
      gpu_context_->opencl_cache_storage(),
      GPUPriorityHint::PRIORITY_NORMAL,
      GPUPerfHint::PERF_HIGH,
      nullptr,
      num_threads,
      cpu_affinity_policy,
      thread_pool_.get());
#endif
}

Device *OpTestContext::GetDevice(DeviceType device_type) {
  return device_map_[device_type].get();
}

#ifdef MACE_ENABLE_OPENCL
std::shared_ptr<GPUContext> OpTestContext::gpu_context() const {
  return gpu_context_;
}

std::vector<MemoryType> OpTestContext::opencl_mem_types() {
  return opencl_mem_types_;
}

void OpTestContext::SetOCLBufferTestFlag() {
  opencl_mem_types_ = {MemoryType::GPU_BUFFER};
}

void OpTestContext::SetOCLImageTestFlag() {
  opencl_mem_types_ = {MemoryType::GPU_IMAGE};
}

void OpTestContext::SetOCLImageAndBufferTestFlag() {
  opencl_mem_types_ = {MemoryType::GPU_IMAGE, MemoryType::GPU_BUFFER};
}
#endif  // MACE_ENABLE_OPENCL

bool OpsTestNet::Setup(mace::DeviceType device) {
  NetDef net_def;
  for (auto &op_def : op_defs_) {
    auto target_op = net_def.add_op();
    target_op->CopyFrom(op_def);

    auto has_data_format = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op_def, "has_data_format", 0);
    auto is_quantized_op = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
        op_def, "T", static_cast<int>(DT_FLOAT))
        == static_cast<int>(DT_UINT8);
    for (auto input : op_def.input()) {
      if (ws_.GetTensor(input) != nullptr &&
          !ws_.GetTensor(input)->is_weight()) {
        auto input_info = net_def.add_input_info();
        input_info->set_name(input);
        if (has_data_format) {
          if (is_quantized_op || device == DeviceType::GPU) {
            input_info->set_data_format(static_cast<int>(DataFormat::NHWC));
          } else {
            input_info->set_data_format(static_cast<int>(DataFormat::NCHW));
          }
        } else {
          input_info->set_data_format(static_cast<int>(DataFormat::NONE));
        }
        auto &shape = ws_.GetTensor(input)->shape();
        for (auto d : shape) {
          input_info->add_dims(static_cast<int>(d));
        }
      }
    }
    if (has_data_format) {
      SetProtoArg<int>(target_op, "data_format",
                       static_cast<int>(DataFormat::AUTO));
    }
  }
  if (!op_defs_.empty()) {
    auto op_def = op_defs_.back();
    for (int i = 0; i < op_def.output_size(); ++i) {
      ws_.RemoveTensor(op_def.output(i));
      auto output_info = net_def.add_output_info();
      output_info->set_name(op_def.output(i));
      if (op_def.output_type_size() == op_def.output_size()) {
        output_info->set_data_type(op_def.output_type(i));
      } else {
        output_info->set_data_type(DataType::DT_FLOAT);
      }
    }
  }
  NetDef adapted_net_def;
  NetDefAdapter net_def_adapter(op_registry_.get(), &ws_);
  net_def_adapter.AdaptNetDef(&net_def,
                              OpTestContext::Get()->GetDevice(device),
                              &adapted_net_def);

  MemoryOptimizer mem_optimizer;
  net_ = make_unique<SerialNet>(
      op_registry_.get(),
      &adapted_net_def,
      &ws_,
      OpTestContext::Get()->GetDevice(device),
      &mem_optimizer);
  MaceStatus status = (ws_.PreallocateOutputTensor(
      adapted_net_def,
      &mem_optimizer,
      OpTestContext::Get()->GetDevice(device)));
  if (status != MaceStatus::MACE_SUCCESS) return false;
  status = net_->Init();
  device_type_ = device;
  return status == MaceStatus::MACE_SUCCESS;
}

MaceStatus OpsTestNet::Run() {
  MACE_CHECK_NOTNULL(net_);
  MACE_RETURN_IF_ERROR(net_->Run());
  Sync();
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpsTestNet::RunOp(mace::DeviceType device) {
  if (device == DeviceType::GPU) {
#ifdef MACE_ENABLE_OPENCL
    auto opencl_mem_types = OpTestContext::Get()->opencl_mem_types();
    for (auto type : opencl_mem_types) {
      OpTestContext::Get()->GetDevice(device)
          ->gpu_runtime()->set_mem_type(type);
      Setup(device);
      MACE_RETURN_IF_ERROR(Run());
    }
    return MaceStatus::MACE_SUCCESS;
#else
    return MaceStatus::MACE_UNSUPPORTED;
#endif  // MACE_ENABLE_OPENCL
  } else {
    Setup(device);
    return Run();
  }
}

MaceStatus OpsTestNet::RunOp() {
  return RunOp(DeviceType::CPU);
}

MaceStatus OpsTestNet::RunNet(const mace::NetDef &net_def,
                              const mace::DeviceType device) {
  device_type_ = device;
  NetDef adapted_net_def;
  NetDefAdapter net_def_adapter(op_registry_.get(), &ws_);
  net_def_adapter.AdaptNetDef(&net_def,
                              OpTestContext::Get()->GetDevice(device),
                              &adapted_net_def);
  MemoryOptimizer mem_optimizer;
  net_ = make_unique<SerialNet>(
      op_registry_.get(),
      &adapted_net_def,
      &ws_,
      OpTestContext::Get()->GetDevice(device),
      &mem_optimizer);
  MACE_RETURN_IF_ERROR(ws_.PreallocateOutputTensor(
      adapted_net_def,
      &mem_optimizer,
      OpTestContext::Get()->GetDevice(device)));
  MACE_RETURN_IF_ERROR(net_->Init());
  return net_->Run();
}

void OpsTestNet::Sync() {
#ifdef MACE_ENABLE_OPENCL
  if (net_ && device_type_ == DeviceType::GPU) {
      OpTestContext::Get()->GetDevice(DeviceType::GPU)->gpu_runtime()
      ->opencl_runtime()->command_queue().finish();
    }
#endif
}

}  // namespace test
}  // namespace ops
}  // namespace mace

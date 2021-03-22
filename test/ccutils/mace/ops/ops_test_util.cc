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

#include <sys/stat.h>

#include "mace/core/memory/rpcmem/rpcmem.h"
#include "mace/core/net_def_adapter.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/runtimes/opencl/opencl_runtime.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

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
std::string GetStoragePathFromEnv() {
  char *storage_path_str = getenv("MACE_INTERNAL_STORAGE_PATH");
  if (storage_path_str == nullptr) return "";
  return storage_path_str;
}
}  // namespace

OpTestContext *OpTestContext::Get(int num_threads,
                                  CPUAffinityPolicy cpu_affinity_policy) {
  static OpTestContext instance(num_threads,
                                cpu_affinity_policy);
  return &instance;
}

std::unique_ptr<OpTestContext> OpTestContext::New(
    int num_threads, CPUAffinityPolicy cpu_affinity_policy) {
  return make_unique<OpTestContext>(num_threads, cpu_affinity_policy, false);
}


std::string GetStoragePath(bool default_path) {
  auto storage_path = GetStoragePathFromEnv();
  if (!default_path) {
    static int cache_count = 0;
    cache_count++;
    std::ostringstream os;
    os << storage_path << "/" << cache_count;
    storage_path = os.str();
    mkdir(storage_path.c_str(), S_IRWXU);
  }
  return storage_path;
}

OpTestContext::OpTestContext(int num_threads,
                             CPUAffinityPolicy cpu_affinity_policy,
                             bool default_path) :
#ifdef MACE_ENABLE_OPENCL
    gpu_context_(std::make_shared<OpenclContext>(GetStoragePath(default_path))),
    opencl_mem_types_({MemoryType::GPU_IMAGE}),
#endif
    thread_pool_(make_unique<utils::ThreadPool>(num_threads,
                                                cpu_affinity_policy)),
    rpcmem_(rpcmem_factory::CreateRpcmem()),
    runtime_context_(new IonRuntimeContext(thread_pool_.get(), rpcmem_)),
    runtime_registry_(new RuntimeRegistry) {
  thread_pool_->Init();
  RegisterAllRuntimes(runtime_registry_.get());

  MaceEngineCfgImpl engine_config;
  engine_config.SetCPUThreadPolicy(num_threads, cpu_affinity_policy);
  auto cpu_runtime = SmartCreateRuntime(
      runtime_registry_.get(), RuntimeType::RT_CPU, runtime_context_.get());
  cpu_runtime->Init(&engine_config, CPU_BUFFER);
  runtime_map_[RuntimeType::RT_CPU] = std::move(cpu_runtime);

#ifdef MACE_ENABLE_OPENCL
  engine_config.SetOpenclContext(gpu_context_);
  engine_config.SetGPUHints(GPUPerfHint::PERF_HIGH,
                            GPUPriorityHint::PRIORITY_NORMAL);
  auto opencl_runtime = SmartCreateRuntime(
      runtime_registry_.get(), RuntimeType::RT_OPENCL, runtime_context_.get());
  opencl_runtime->Init(&engine_config, GPU_IMAGE);
  runtime_map_[RuntimeType::RT_OPENCL] = std::move(opencl_runtime);
#endif
}

std::unique_ptr<Runtime> OpTestContext::NewAndInitRuntime(
    RuntimeType runtime_type, MemoryType mem_type,
    MaceEngineCfgImpl *engine_config, RuntimeContext *runtime_context) {
  if (runtime_context == nullptr) {
    runtime_context = runtime_context_.get();
  }

  auto runtime = SmartCreateRuntime(
      runtime_registry_.get(), runtime_type, runtime_context);
  runtime->Init(engine_config, mem_type);

  return runtime;
}

Runtime *OpTestContext::GetRuntime(RuntimeType runtime_type) {
  return runtime_map_[runtime_type].get();
}

#ifdef MACE_ENABLE_OPENCL
std::shared_ptr<OpenclContext> OpTestContext::gpu_context() const {
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


int OpsTestNet::ref_count_ = 0;
std::mutex OpsTestNet::ref_mutex_;
bool OpsTestNet::Setup(mace::RuntimeType runtime_type) {
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
          if (is_quantized_op || runtime_type == RuntimeType::RT_OPENCL) {
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
  auto *cpu_runtime = OpTestContext::Get()->GetRuntime(RuntimeType::RT_CPU);
  auto *target_runtime = OpTestContext::Get()->GetRuntime(runtime_type);
  net_def_adapter.AdaptNetDef(&net_def, target_runtime,
                              cpu_runtime, &adapted_net_def);

  net_ = make_unique<SerialNet>(op_registry_.get(), &adapted_net_def, &ws_,
                                target_runtime, cpu_runtime);
  MaceStatus status = net_->Init();
  runtime_type_ = runtime_type;
  return status == MaceStatus::MACE_SUCCESS;
}

MaceStatus OpsTestNet::Run() {
  MACE_CHECK_NOTNULL(net_);
  MACE_RETURN_IF_ERROR(net_->Run());
  Sync();
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpsTestNet::RunOp(mace::RuntimeType runtime_type) {
  if (runtime_type == RuntimeType::RT_OPENCL) {
#ifdef MACE_ENABLE_OPENCL
    auto opencl_mem_types = OpTestContext::Get()->opencl_mem_types();
    for (auto type : opencl_mem_types) {
      auto* runtime = OpTestContext::Get()->GetRuntime(runtime_type);
      auto *opencl_runtime = static_cast<OpenclRuntime *>(runtime);
      opencl_runtime->SetUsedMemoryType(type);
      Setup(runtime_type);
      MACE_RETURN_IF_ERROR(Run());
    }
    return MaceStatus::MACE_SUCCESS;
#else
    return MaceStatus::MACE_UNSUPPORTED;
#endif  // MACE_ENABLE_OPENCL
  } else {
    Setup(runtime_type);
    return Run();
  }
}

MaceStatus OpsTestNet::RunOp() {
  return RunOp(RuntimeType::RT_CPU);
}

MaceStatus OpsTestNet::RunNet(const mace::NetDef &net_def,
                              const mace::RuntimeType runtime_type) {
  runtime_type_ = runtime_type;
  NetDef adapted_net_def;
  NetDefAdapter net_def_adapter(op_registry_.get(), &ws_);
  auto *cpu_runtime = OpTestContext::Get()->GetRuntime(RuntimeType::RT_CPU);
  auto *target_runtime = OpTestContext::Get()->GetRuntime(runtime_type);
  net_def_adapter.AdaptNetDef(&net_def, target_runtime,
                              cpu_runtime, &adapted_net_def);

  net_ = make_unique<SerialNet>(op_registry_.get(), &adapted_net_def, &ws_,
                                target_runtime, cpu_runtime);
  MACE_RETURN_IF_ERROR(net_->Init());
  return net_->Run();
}

void OpsTestNet::Sync() {
#ifdef MACE_ENABLE_OPENCL
  if (net_ && runtime_type_ == RuntimeType::RT_OPENCL) {
    auto *runtime = OpTestContext::Get()->GetRuntime(RuntimeType::RT_OPENCL);
    auto *opencl_runtime = static_cast<OpenclRuntime *>(runtime);
    opencl_runtime->GetOpenclExecutor()->command_queue().finish();
  }
#endif
}

OpsTestNet::~OpsTestNet() {
  std::lock_guard<std::mutex> lock(ref_mutex_);
  --ref_count_;
  if (ref_count_ == 0) {
    auto *cpu_runtime = OpTestContext::Get()->GetRuntime(RT_CPU);
    cpu_runtime->ReleaseAllBuffer(RENT_SHARE, true);
    cpu_runtime->ReleaseAllBuffer(RENT_PRIVATE, true);
#ifdef MACE_ENABLE_OPENCL
    auto *target_runtime = OpTestContext::Get()->GetRuntime(RT_OPENCL);
    target_runtime->ReleaseAllBuffer(RENT_SHARE, true);
    target_runtime->ReleaseAllBuffer(RENT_PRIVATE, true);
#endif  // MACE_ENABLE_OPENCL
  }
}

}  // namespace test
}  // namespace ops
}  // namespace mace

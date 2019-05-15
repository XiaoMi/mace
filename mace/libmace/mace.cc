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

#include <algorithm>
#include <numeric>
#include <memory>

#include "mace/core/device_context.h"
#include "mace/core/memory_optimizer.h"
#include "mace/core/net.h"
#include "mace/ops/registry/ops_registry.h"
#include "mace/ops/common/transpose.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"
#include "mace/utils/stl_util.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/core/net_def_adapter.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/gpu_device.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#endif  // MACE_ENABLE_OPENCL

#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
#include "mace/core/runtime/hexagon/hexagon_device.h"
#endif

#ifdef MACE_ENABLE_APU
#include "mace/core/runtime/apu/apu_wrapper.h"
#include "mace/core/runtime/apu/apu_device.h"
#endif  // MACE_ENABLE_APU

namespace mace {

namespace {

#ifdef MACE_ENABLE_OPENCL
MaceStatus CheckGPUAvalibility(const NetDef *net_def, Device *device) {
  // Check OpenCL avaliable
  auto runtime = device->gpu_runtime();
  if (!runtime->opencl_runtime()->is_opencl_avaliable()) {
    LOG(WARNING) << "The device does not support OpenCL";
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  // Check whether model max OpenCL image sizes exceed OpenCL limitation.
  if (net_def == nullptr) {
    return MaceStatus::MACE_INVALID_ARGS;
  }

  const int mem_type_i =
      ProtoArgHelper::GetOptionalArg<NetDef, int>(
          *net_def, "opencl_mem_type",
          static_cast<MemoryType>(MemoryType::GPU_IMAGE));
  const MemoryType mem_type = static_cast<MemoryType>(mem_type_i);

  runtime->set_mem_type(mem_type);

  return MaceStatus::MACE_SUCCESS;
}
#endif

}  // namespace

class GPUContextBuilder::Impl {
 public:
  Impl();
  void SetStoragePath(const std::string &path);

  void SetOpenCLBinaryPaths(const std::vector<std::string> &paths);

  void SetOpenCLBinary(const unsigned char *data, const size_t size);

  void SetOpenCLParameterPath(const std::string &path);

  void SetOpenCLParameter(const unsigned char *data, const size_t size);

  std::shared_ptr<GPUContext> Finalize();

 public:
  std::string storage_path_;
  std::vector<std::string> opencl_binary_paths_;
  std::string opencl_parameter_path_;
  const unsigned char *opencl_binary_ptr_;
  size_t opencl_binary_size_;
  const unsigned char *opencl_parameter_ptr_;
  size_t opencl_parameter_size_;
};

GPUContextBuilder::Impl::Impl()
    : storage_path_(""), opencl_binary_paths_(0), opencl_parameter_path_(""),
      opencl_binary_ptr_(nullptr), opencl_binary_size_(0),
      opencl_parameter_ptr_(nullptr), opencl_parameter_size_(0) {}

void GPUContextBuilder::Impl::SetStoragePath(const std::string &path) {
  storage_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  opencl_binary_paths_ = paths;
}

void GPUContextBuilder::Impl::SetOpenCLBinary(const unsigned char *data,
                                              const size_t size) {
  opencl_binary_ptr_ = data;
  opencl_binary_size_ = size;
}

void GPUContextBuilder::Impl::SetOpenCLParameterPath(
    const std::string &path) {
  opencl_parameter_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLParameter(const unsigned char *data,
                                                 const size_t size) {
  opencl_parameter_ptr_ = data;
  opencl_parameter_size_ = size;
}

std::shared_ptr<GPUContext> GPUContextBuilder::Impl::Finalize() {
  return std::shared_ptr<GPUContext>(new GPUContext(storage_path_,
                                                    opencl_binary_paths_,
                                                    opencl_parameter_path_,
                                                    opencl_binary_ptr_,
                                                    opencl_binary_size_,
                                                    opencl_parameter_ptr_,
                                                    opencl_parameter_size_));
}

GPUContextBuilder::GPUContextBuilder() : impl_(new GPUContextBuilder::Impl) {}

GPUContextBuilder::~GPUContextBuilder() = default;

GPUContextBuilder &GPUContextBuilder::SetStoragePath(const std::string &path) {
  impl_->SetStoragePath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  impl_->SetOpenCLBinaryPaths(paths);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinary(
    const unsigned char *data, const size_t size) {
  impl_->SetOpenCLBinary(data, size);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameterPath(
    const std::string &path) {
  impl_->SetOpenCLParameterPath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameter(
    const unsigned char *data, const size_t size) {
  impl_->SetOpenCLParameter(data, size);
  return *this;
}

std::shared_ptr<GPUContext> GPUContextBuilder::Finalize() {
  return impl_->Finalize();
}

class MaceEngineConfig::Impl {
 public:
  explicit Impl(const DeviceType device_type);
  ~Impl() = default;

  MaceStatus SetGPUContext(std::shared_ptr<GPUContext> context);

  MaceStatus SetGPUHints(GPUPerfHint perf_hint, GPUPriorityHint priority_hint);

  MaceStatus SetCPUThreadPolicy(int num_threads_hint,
                                CPUAffinityPolicy policy);

  inline DeviceType device_type() const {
    return device_type_;
  }

  inline int num_threads() const {
    return num_threads_;
  }

  inline CPUAffinityPolicy cpu_affinity_policy() const {
    return cpu_affinity_policy_;
  }

  inline std::shared_ptr<GPUContext> gpu_context() const {
    return gpu_context_;
  }

  inline GPUPriorityHint gpu_priority_hint() const {
    return gpu_priority_hint_;
  }

  inline GPUPerfHint gpu_perf_hint() const {
    return gpu_perf_hint_;
  }

 private:
  DeviceType device_type_;
  int num_threads_;
  CPUAffinityPolicy cpu_affinity_policy_;
  std::shared_ptr<GPUContext> gpu_context_;
  GPUPriorityHint gpu_priority_hint_;
  GPUPerfHint gpu_perf_hint_;
};

MaceEngineConfig::Impl::Impl(const DeviceType device_type)
    : device_type_(device_type),
      num_threads_(-1),
      cpu_affinity_policy_(CPUAffinityPolicy::AFFINITY_NONE),
      gpu_context_(new GPUContext),
      gpu_priority_hint_(GPUPriorityHint::PRIORITY_LOW),
      gpu_perf_hint_(GPUPerfHint::PERF_NORMAL) {}

MaceStatus MaceEngineConfig::Impl::SetGPUContext(
    std::shared_ptr<GPUContext> context) {
  gpu_context_ = context;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngineConfig::Impl::SetGPUHints(
    GPUPerfHint perf_hint,
    GPUPriorityHint priority_hint) {
  gpu_perf_hint_ = perf_hint;
  gpu_priority_hint_ = priority_hint;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngineConfig::Impl::SetCPUThreadPolicy(
    int num_threads,
    CPUAffinityPolicy policy) {
  num_threads_ = num_threads;
  cpu_affinity_policy_ = policy;
  return MaceStatus::MACE_SUCCESS;
}

MaceEngineConfig::MaceEngineConfig(
    const DeviceType device_type)
    : impl_(new MaceEngineConfig::Impl(device_type)) {}

MaceEngineConfig::~MaceEngineConfig() = default;

MaceStatus MaceEngineConfig::SetGPUContext(
    std::shared_ptr<GPUContext> context) {
  return impl_->SetGPUContext(context);
}

MaceStatus MaceEngineConfig::SetGPUHints(
    GPUPerfHint perf_hint,
    GPUPriorityHint priority_hint) {
  return impl_->SetGPUHints(perf_hint, priority_hint);
}

MaceStatus MaceEngineConfig::SetCPUThreadPolicy(
    int num_threads_hint,
    CPUAffinityPolicy policy) {
  return impl_->SetCPUThreadPolicy(num_threads_hint, policy);
}

// Mace Tensor
class MaceTensor::Impl {
 public:
  std::vector<int64_t> shape;
  std::shared_ptr<void> data;
  DataFormat format;
  int64_t buffer_size;
};

MaceTensor::MaceTensor(const std::vector<int64_t> &shape,
                       std::shared_ptr<void> data,
                       const DataFormat format) {
  MACE_CHECK_NOTNULL(data.get());
  MACE_CHECK(format == DataFormat::NONE || format == DataFormat::NHWC
                 || format == DataFormat::NCHW || format == DataFormat::OIHW,
             "MACE only support NONE, NHWC, NCHW and OIHW "
             "formats of input now.");
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = shape;
  impl_->data = data;
  impl_->format = format;
  impl_->buffer_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>());
}

MaceTensor::MaceTensor() {
  impl_ = make_unique<MaceTensor::Impl>();
}

MaceTensor::MaceTensor(const MaceTensor &other) {
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
}

MaceTensor::MaceTensor(const MaceTensor &&other) {
  impl_ = make_unique<MaceTensor::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
}

MaceTensor &MaceTensor::operator=(const MaceTensor &other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

MaceTensor &MaceTensor::operator=(const MaceTensor &&other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

MaceTensor::~MaceTensor() = default;

const std::vector<int64_t> &MaceTensor::shape() const { return impl_->shape; }

const std::shared_ptr<float> MaceTensor::data() const {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<float> MaceTensor::data() {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<void> MaceTensor::raw_data() const {
  return impl_->data;
}

std::shared_ptr<void> MaceTensor::raw_mutable_data() {
  return impl_->data;
}

DataFormat MaceTensor::data_format() const {
  return impl_->format;
}

// Mace Engine
class MaceEngine::Impl {
 public:
  explicit Impl(const MaceEngineConfig &config);

  ~Impl();

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data);

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const std::string &model_data_file);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata);

 private:
  MaceStatus TransposeInput(
      const std::pair<const std::string, MaceTensor> &input,
      Tensor *input_tensor);

  MaceStatus TransposeOutput(const Tensor *output_tensor,
                             std::pair<const std::string, MaceTensor> *output);

 private:
  std::unique_ptr<port::ReadOnlyMemoryRegion> model_data_;
  std::unique_ptr<OpRegistryBase> op_registry_;
  DeviceType device_type_;
  std::unique_ptr<Device> device_;
  std::unique_ptr<Workspace> ws_;
  std::unique_ptr<NetBase> net_;
  bool is_quantized_model_;
  std::map<std::string, mace::InputOutputInfo> input_info_map_;
  std::map<std::string, mace::InputOutputInfo> output_info_map_;
  std::unique_ptr<utils::ThreadPool> thread_pool_;
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  std::unique_ptr<HexagonControlWrapper> hexagon_controller_;
#endif
#ifdef MACE_ENABLE_APU
  std::unique_ptr<ApuWrapper> apu_controller_;
#endif

  MACE_DISABLE_COPY_AND_ASSIGN(Impl);
};

MaceEngine::Impl::Impl(const MaceEngineConfig &config)
    : model_data_(nullptr),
      op_registry_(new OpRegistry),
      device_type_(config.impl_->device_type()),
      device_(nullptr),
      ws_(new Workspace()),
      net_(nullptr),
      is_quantized_model_(false),
      thread_pool_(new utils::ThreadPool(config.impl_->num_threads(),
                                         config.impl_->cpu_affinity_policy()))
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
, hexagon_controller_(nullptr)
#endif
#ifdef MACE_ENABLE_APU
      , apu_controller_(nullptr)
#endif
{
  LOG(INFO) << "Creating MaceEngine, MACE version: " << MaceVersion();
  thread_pool_->Init();
  if (device_type_ == DeviceType::CPU) {
    device_.reset(new CPUDevice(config.impl_->num_threads(),
                                config.impl_->cpu_affinity_policy(),
                                thread_pool_.get()));
  }
#ifdef MACE_ENABLE_OPENCL
  if (device_type_ == DeviceType::GPU) {
    device_.reset(new GPUDevice(
        config.impl_->gpu_context()->opencl_tuner(),
        config.impl_->gpu_context()->opencl_cache_storage(),
        config.impl_->gpu_priority_hint(),
        config.impl_->gpu_perf_hint(),
        config.impl_->gpu_context()->opencl_binary_storage(),
        config.impl_->num_threads(),
        config.impl_->cpu_affinity_policy(),
        thread_pool_.get()));
  }
#endif
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  if (device_type_ == DeviceType::HEXAGON
      || device_type_ == DeviceType::HTA) {
    device_.reset(new HexagonDevice(device_type_, thread_pool_.get()));
  }
#endif
#ifdef MACE_ENABLE_APU
  if (device_type_ == DeviceType::APU) {
    device_.reset(new ApuDevice(thread_pool_.get()));
  }
#endif
  MACE_CHECK_NOTNULL(device_);
}

MaceStatus MaceEngine::Impl::Init(
    const NetDef *net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data) {
  LOG(INFO) << "Initializing MaceEngine";
  // Check avalibility
#ifdef MACE_ENABLE_OPENCL
  if (device_type_ == DeviceType::GPU) {
    MACE_RETURN_IF_ERROR(CheckGPUAvalibility(net_def, device_.get()));
  }
#endif
  // mark quantized model flag
  is_quantized_model_ = IsQuantizedModel(*net_def);
  // Get input and output information.
  for (auto &input_info : net_def->input_info()) {
    input_info_map_[input_info.name()] = input_info;
  }
  for (auto &output_info : net_def->output_info()) {
    output_info_map_[output_info.name()] = output_info;
  }
  // Set storage path for internal usage
  for (auto input_name : input_nodes) {
    if (input_info_map_.find(input_name) == input_info_map_.end()) {
      LOG(FATAL) << "'" << input_name
                 << "' does not belong to model's inputs: "
                 << MakeString(MapKeys(input_info_map_));
    }
    DataType input_dt = input_info_map_[input_name].data_type();
    Tensor *input_tensor =
        ws_->CreateTensor(input_name, device_->allocator(), input_dt);
    // Resize to possible largest shape to avoid resize during running.
    std::vector<index_t> shape(input_info_map_[input_name].dims_size());
    for (int i = 0; i < input_info_map_[input_name].dims_size(); ++i) {
      shape[i] = input_info_map_[input_name].dims(i);
    }
    input_tensor->Resize(shape);
    // Set to the default data format
    input_tensor->set_data_format(static_cast<DataFormat>(
        input_info_map_[input_name].data_format()));
  }
  for (auto output_name : output_nodes) {
    if (output_info_map_.find(output_name) == output_info_map_.end()) {
      LOG(FATAL) << "'" << output_name
                 << "' does not belong to model's outputs "
                 << MakeString(MapKeys(output_info_map_));
    }
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
    DataType output_dt = output_info_map_[output_name].data_type();
    Tensor *output_tensor =
        ws_->CreateTensor(output_name, device_->allocator(), output_dt);
    output_tensor->set_data_format(DataFormat::NHWC);
#endif
#if defined(MACE_ENABLE_APU)
    Tensor *output_tensor =
        ws_->CreateTensor(output_name, device_->allocator(), DT_FLOAT);
    output_tensor->set_data_format(DataFormat::NHWC);
#endif
  }
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  if (device_type_ == HEXAGON || device_type_ == HTA) {
    hexagon_controller_ = CreateHexagonControlWrapper(device_.get());
    MACE_CHECK(hexagon_controller_->Config(), "hexagon config error");
    MACE_CHECK(hexagon_controller_->Init(), "hexagon init error");
    hexagon_controller_->SetDebugLevel(
        static_cast<int>(mace::port::MinVLogLevelFromEnv()));
    MACE_CHECK(hexagon_controller_->SetupGraph(*net_def, model_data),
               "hexagon setup graph error");
    if (VLOG_IS_ON(2)) {
      hexagon_controller_->PrintGraph();
    }
  } else {
#endif
#ifdef MACE_ENABLE_APU
  if (device_type_ == APU) {
    apu_controller_.reset(new ApuWrapper(device_.get()));
    MACE_CHECK(apu_controller_->Init(*net_def, model_data), "apu init error");
  } else {
#endif
    MACE_RETURN_IF_ERROR(ws_->LoadModelTensor(*net_def,
                                              device_.get(),
                                              model_data));

    NetDef adapted_net_def;
    NetDefAdapter net_def_adapter(op_registry_.get(), ws_.get());
    net_def_adapter.AdaptNetDef(net_def, device_.get(), &adapted_net_def);

    MemoryOptimizer mem_optimizer;
    // Init model
    net_ = std::unique_ptr<NetBase>(new SerialNet(op_registry_.get(),
                                                  &adapted_net_def,
                                                  ws_.get(),
                                                  device_.get(),
                                                  &mem_optimizer));

    // Preallocate all output tensors of ops
    MACE_RETURN_IF_ERROR(ws_->PreallocateOutputTensor(adapted_net_def,
                                                      &mem_optimizer,
                                                      device_.get()));
    if (device_type_ == DeviceType::GPU) {
      ws_->RemoveAndReloadBuffer(adapted_net_def,
                                 model_data,
                                 device_->allocator());
    }
    MACE_RETURN_IF_ERROR(net_->Init());
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  }
#endif
#ifdef MACE_ENABLE_APU
  }
#endif

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngine::Impl::Init(
    const NetDef *net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const std::string &model_data_file) {
  LOG(INFO) << "Loading Model Data";

  auto fs = GetFileSystem();
  MACE_RETURN_IF_ERROR(fs->NewReadOnlyMemoryRegionFromFile(
      model_data_file.c_str(), &model_data_));

  MACE_RETURN_IF_ERROR(Init(net_def, input_nodes, output_nodes,
      reinterpret_cast<const unsigned char *>(model_data_->data())));

  if (device_type_ == DeviceType::GPU || device_type_ == DeviceType::HEXAGON ||
      device_type_ == DeviceType::HTA ||
      (device_type_ == DeviceType::CPU && ws_->diffused_buffer())) {
    model_data_.reset();
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceEngine::Impl::~Impl() {
  LOG(INFO) << "Destroying MaceEngine";
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  if (device_type_ == HEXAGON || device_type_ == HTA) {
    if (VLOG_IS_ON(2)) {
      hexagon_controller_->GetPerfInfo();
      hexagon_controller_->PrintLog();
    }
    MACE_CHECK(hexagon_controller_->TeardownGraph(), "hexagon teardown error");
    MACE_CHECK(hexagon_controller_->Finalize(), "hexagon finalize error");
  }
#endif
#ifdef MACE_ENABLE_APU
  if (device_type_ == APU) {
    MACE_CHECK(apu_controller_->Uninit(), "apu uninit error");
  }
#endif
}

MaceStatus MaceEngine::Impl::TransposeInput(
    const std::pair<const std::string, MaceTensor> &input,
    Tensor *input_tensor) {
  bool has_data_format = input_tensor->data_format() != DataFormat::NONE;
  DataFormat data_format = DataFormat::NONE;
  DataType input_dt = input_tensor->dtype();
  if (has_data_format) {
    std::vector<int> dst_dims;
    if (device_->device_type() == DeviceType::CPU &&
        input.second.shape().size() == 4 &&
        input.second.data_format() == DataFormat::NHWC &&
        !is_quantized_model_) {
      VLOG(1) << "Transform input " << input.first << " from NHWC to NCHW";
      input_tensor->set_data_format(DataFormat::NCHW);
      dst_dims = {0, 3, 1, 2};
    } else if (
        (is_quantized_model_ || device_->device_type() == DeviceType::GPU) &&
            input.second.shape().size() == 4 &&
            input.second.data_format() == DataFormat::NCHW) {
      VLOG(1) << "Transform input " << input.first << " from NCHW to NHWC";
      input_tensor->set_data_format(DataFormat::NHWC);
      dst_dims = {0, 2, 3, 1};
    }
    if (!dst_dims.empty()) {
      std::vector<index_t> output_shape =
          TransposeShape<int64_t, index_t>(input.second.shape(), dst_dims);
      MACE_RETURN_IF_ERROR(input_tensor->Resize(output_shape));
      Tensor::MappingGuard input_guard(input_tensor);
      if (input_dt == DataType::DT_FLOAT) {
        auto input_data = input_tensor->mutable_data<float>();
        return ops::Transpose(thread_pool_.get(),
                              input.second.data<float>().get(),
                              input.second.shape(),
                              dst_dims,
                              input_data);
      } else if (input_dt == DataType::DT_INT32) {
        auto input_data = input_tensor->mutable_data<int>();
        return ops::Transpose(thread_pool_.get(),
                              input.second.data<int>().get(),
                              input.second.shape(),
                              dst_dims,
                              input_data);
      } else {
        LOG(FATAL) << "MACE do not support the input data type: " << input_dt;
      }
    }

    data_format = input.second.data_format();
  }
  input_tensor->set_data_format(data_format);
  MACE_RETURN_IF_ERROR(input_tensor->Resize(input.second.shape()));
  Tensor::MappingGuard input_guard(input_tensor);
  if (input_dt == DataType::DT_FLOAT) {
    auto input_data = input_tensor->mutable_data<float>();
    memcpy(input_data, input.second.data().get(),
           input_tensor->size() * sizeof(float));
  } else if (input_dt == DataType::DT_INT32) {
    auto input_data = input_tensor->mutable_data<int>();
    memcpy(input_data, input.second.data().get(),
           input_tensor->size() * sizeof(int));
  } else {
    LOG(FATAL) << "MACE do not support the input data type: " << input_dt;
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngine::Impl::TransposeOutput(
    const mace::Tensor *output_tensor,
    std::pair<const std::string, mace::MaceTensor> *output) {
  DataType output_dt = output_tensor->dtype();
  // save output
  if (output_tensor != nullptr && output->second.data() != nullptr) {
    if (output_tensor->data_format() != DataFormat::NONE &&
        output->second.data_format() != DataFormat::NONE &&
        output->second.shape().size() == 4 &&
        output->second.data_format() != output_tensor->data_format()) {
      VLOG(1) << "Transform output " << output->first << " from "
              << static_cast<int>(output_tensor->data_format()) << " to "
              << static_cast<int>(output->second.data_format());
      std::vector<int> dst_dims;
      if (output_tensor->data_format() == DataFormat::NCHW &&
          output->second.data_format() == DataFormat::NHWC) {
        dst_dims = {0, 2, 3, 1};
      } else if (output_tensor->data_format() == DataFormat::NHWC &&
          output->second.data_format() == DataFormat::NCHW) {
        dst_dims = {0, 3, 1, 2};
      } else {
        LOG(FATAL) << "Not supported output data format: "
                   << static_cast<int>(output->second.data_format()) << " vs "
                   << static_cast<int>(output_tensor->data_format());
      }
      VLOG(1) << "Transform output " << output->first << " from "
              << static_cast<int>(output_tensor->data_format()) << " to "
              << static_cast<int>(output->second.data_format());
      std::vector<index_t> shape =
          TransposeShape<index_t, index_t>(output_tensor->shape(),
                                           dst_dims);
      int64_t output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<int64_t>());
      MACE_CHECK(output_size <= output->second.impl_->buffer_size)
        << "Output size exceeds buffer size: shape"
        << MakeString<int64_t>(shape) << " vs buffer size "
        << output->second.impl_->buffer_size;
      output->second.impl_->shape = shape;
      Tensor::MappingGuard output_guard(output_tensor);
      if (output_dt == DataType::DT_FLOAT) {
        auto output_data = output_tensor->data<float>();
        return ops::Transpose(thread_pool_.get(),
                              output_data,
                              output_tensor->shape(),
                              dst_dims,
                              output->second.data<float>().get());
      } else if (output_dt == DataType::DT_INT32) {
        auto output_data = output_tensor->data<int>();
        return ops::Transpose(thread_pool_.get(),
                              output_data,
                              output_tensor->shape(),
                              dst_dims,
                              output->second.data<int>().get());
      } else {
        LOG(FATAL) << "MACE do not support the output data type: " << output_dt;
        return MaceStatus::MACE_INVALID_ARGS;
      }
    } else {
      Tensor::MappingGuard output_guard(output_tensor);
      auto shape = output_tensor->shape();
      int64_t output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<int64_t>());
      MACE_CHECK(output_size <= output->second.impl_->buffer_size)
        << "Output size exceeds buffer size: shape"
        << MakeString<int64_t>(shape) << " vs buffer size "
        << output->second.impl_->buffer_size;
      output->second.impl_->shape = shape;
      if (output_dt == DataType::DT_FLOAT) {
        std::memcpy(output->second.data<float>().get(),
                    output_tensor->data<float>(),
                    output_size * sizeof(float));
      } else if (output_dt == DataType::DT_INT32) {
        std::memcpy(output->second.data<int>().get(),
                    output_tensor->data<int>(),
                    output_size * sizeof(int));
      } else {
        LOG(FATAL) << "MACE do not support the output data type: " << output_dt;
      }
      return MaceStatus::MACE_SUCCESS;
    }
  } else {
    return MaceStatus::MACE_INVALID_ARGS;
  }
}

MaceStatus MaceEngine::Impl::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  MACE_CHECK_NOTNULL(outputs);
  std::map<std::string, Tensor *> input_tensors;
  std::map<std::string, Tensor *> output_tensors;
  for (auto &input : inputs) {
    if (input_info_map_.find(input.first) == input_info_map_.end()) {
      LOG(FATAL) << "'" << input.first
                 << "' does not belong to model's inputs: "
                 << MakeString(MapKeys(input_info_map_));
    }
    Tensor *input_tensor = ws_->GetTensor(input.first);
    MACE_RETURN_IF_ERROR(TransposeInput(input, input_tensor));
    input_tensors[input.first] = input_tensor;
  }
  for (auto &output : *outputs) {
    if (output_info_map_.find(output.first) == output_info_map_.end()) {
      LOG(FATAL) << "'" << output.first
                 << "' does not belong to model's outputs: "
                 << MakeString(MapKeys(output_info_map_));
    }
    Tensor *output_tensor = ws_->GetTensor(output.first);
    output_tensors[output.first] = output_tensor;
  }
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  if (device_type_ == HEXAGON || device_type_ == HTA) {
    if (device_type_ == HTA) {
      MACE_CHECK(input_tensors.size() == 1 && output_tensors.size() == 1,
                 "HTA not support multiple inputs and outputs yet.");
    }
    hexagon_controller_->ExecuteGraphNew(input_tensors, &output_tensors);
  } else {
#endif
#ifdef MACE_ENABLE_APU
  if (device_type_ == APU) {
    MACE_CHECK(apu_controller_->Run(input_tensors, &output_tensors),
               "apu run error");
  } else {
#endif
  MACE_RETURN_IF_ERROR(net_->Run(run_metadata));
#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  }
#endif
#ifdef MACE_ENABLE_APU
  }
#endif

#ifdef MACE_ENABLE_OPENCL
  if (device_type_ == GPU) {
    device_->gpu_runtime()->opencl_runtime()->command_queue().finish();
    device_->gpu_runtime()->opencl_runtime()->SaveBuiltCLProgram();
  }
#endif
  for (auto &output : *outputs) {
    Tensor *output_tensor = ws_->GetTensor(output.first);
    // save output
    MACE_RETURN_IF_ERROR(TransposeOutput(output_tensor, &output));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceEngine::MaceEngine(const MaceEngineConfig &config) :
    impl_(make_unique<MaceEngine::Impl>(config)) {}

MaceEngine::~MaceEngine() = default;

MaceStatus MaceEngine::Init(const NetDef *net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const unsigned char *model_data) {
  return impl_->Init(net_def, input_nodes, output_nodes, model_data);
}

MaceStatus MaceEngine::Init(const NetDef *net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const std::string &model_data_file) {
  return impl_->Init(net_def, input_nodes, output_nodes, model_data_file);
}

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs,
                           RunMetadata *run_metadata) {
  return impl_->Run(inputs, outputs, run_metadata);
}

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs) {
  return impl_->Run(inputs, outputs, nullptr);
}

MaceStatus CreateMaceEngineFromProto(
    const unsigned char *model_graph_proto,
    const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine) {
  // TODO(heliangliang) Add buffer range checking
  MACE_UNUSED(model_weights_data_size);
  LOG(INFO) << "Create MaceEngine from model graph proto and weights data";

  if (engine == nullptr) {
    return MaceStatus::MACE_INVALID_ARGS;
  }

  auto net_def = std::make_shared<NetDef>();
  net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);

  engine->reset(new mace::MaceEngine(config));
  MaceStatus status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_weights_data);

  return status;
}

// Deprecated, will be removed in future version.
MaceStatus CreateMaceEngineFromProto(
    const std::vector<unsigned char> &model_pb,
    const std::string &model_data_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine) {
  LOG(INFO) << "Create MaceEngine from model pb";
  LOG(WARNING) << "Function deprecated, please change to the new API";
  // load model
  if (engine == nullptr) {
    return MaceStatus::MACE_INVALID_ARGS;
  }

  std::shared_ptr<NetDef> net_def(new NetDef());
  net_def->ParseFromArray(&model_pb[0], model_pb.size());

  engine->reset(new mace::MaceEngine(config));
  MaceStatus status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_data_file);

  return status;
}

}  // namespace mace

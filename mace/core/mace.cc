// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/core/file_storage.h"
#include "mace/core/net.h"
#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"

namespace mace {

// Mace Tensor
struct MaceTensor::Impl {
  std::vector<int64_t> shape;
  std::shared_ptr<float> data;
};

MaceTensor::MaceTensor(const std::vector<int64_t> &shape,
                       std::shared_ptr<float> data) {
  MACE_CHECK_NOTNULL(data.get());
  impl_ = std::unique_ptr<MaceTensor::Impl>(new MaceTensor::Impl());
  impl_->shape = shape;
  impl_->data = data;
}

MaceTensor::MaceTensor() {
  impl_ = std::unique_ptr<MaceTensor::Impl>(new MaceTensor::Impl());
}

MaceTensor::MaceTensor(const MaceTensor &other) {
  impl_ = std::unique_ptr<MaceTensor::Impl>(new MaceTensor::Impl());
  impl_->shape = other.shape();
  impl_->data = other.data();
}

MaceTensor::MaceTensor(const MaceTensor &&other) {
  impl_ = std::unique_ptr<MaceTensor::Impl>(new MaceTensor::Impl());
  impl_->shape = std::move(other.shape());
  impl_->data = std::move(other.data());
}

MaceTensor &MaceTensor::operator=(const MaceTensor &other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  return *this;
}

MaceTensor &MaceTensor::operator=(const MaceTensor &&other) {
  impl_->shape = std::move(other.shape());
  impl_->data = std::move(other.data());
  return *this;
}

MaceTensor::~MaceTensor() = default;

const std::vector<int64_t> &MaceTensor::shape() const { return impl_->shape; }

const std::shared_ptr<float> MaceTensor::data() const { return impl_->data; }

std::shared_ptr<float> MaceTensor::data() { return impl_->data; }

// Mace Engine
class MaceEngine::Impl {
 public:
  explicit Impl(const NetDef *net_def,
                DeviceType device_type,
                const std::vector<std::string> &input_nodes,
                const std::vector<std::string> &output_nodes);
  ~Impl();

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata);

 private:
  std::shared_ptr<OperatorRegistry> op_registry_;
  DeviceType device_type_;
  std::unique_ptr<Workspace> ws_;
  std::unique_ptr<NetBase> net_;
  std::unique_ptr<HexagonControlWrapper> hexagon_controller_;

  DISABLE_COPY_AND_ASSIGN(Impl);
};

MaceEngine::Impl::Impl(const NetDef *net_def,
                       DeviceType device_type,
                       const std::vector<std::string> &input_nodes,
                       const std::vector<std::string> &output_nodes)
    : op_registry_(new OperatorRegistry()),
      device_type_(device_type),
      ws_(new Workspace()),
      net_(nullptr),
      hexagon_controller_(nullptr) {
  LOG(INFO) << "MACE version: " << MaceVersion();
  // Set storage path for internal usage
  for (auto input_name : input_nodes) {
    ws_->CreateTensor(MakeString("mace_input_node_", input_name, ":0"),
                      GetDeviceAllocator(device_type_), DT_FLOAT);
  }
  for (auto output_name : output_nodes) {
    ws_->CreateTensor(MakeString("mace_output_node_", output_name, ":0"),
                      GetDeviceAllocator(device_type_), DT_FLOAT);
  }
  if (device_type == HEXAGON) {
    hexagon_controller_.reset(new HexagonControlWrapper());
    MACE_CHECK(hexagon_controller_->Config(), "hexagon config error");
    MACE_CHECK(hexagon_controller_->Init(), "hexagon init error");
    hexagon_controller_->SetDebugLevel(
        static_cast<int>(mace::logging::LogMessage::MinVLogLevel()));
    int dsp_mode =
        ArgumentHelper::GetSingleArgument<NetDef, int>(*net_def, "dsp_mode", 0);
    hexagon_controller_->SetGraphMode(dsp_mode);
    MACE_CHECK(hexagon_controller_->SetupGraph(*net_def),
               "hexagon setup graph error");
    if (VLOG_IS_ON(2)) {
      hexagon_controller_->PrintGraph();
    }
  } else {
    ws_->LoadModelTensor(*net_def, device_type);

    // Init model
    auto net = CreateNet(op_registry_, *net_def, ws_.get(), device_type,
                         NetMode::INIT);
    if (!net->Run()) {
      LOG(FATAL) << "Net init run failed";
    }
    net_ = std::move(CreateNet(op_registry_, *net_def, ws_.get(), device_type));
  }
}

MaceEngine::Impl::~Impl() {
  if (device_type_ == HEXAGON) {
    if (VLOG_IS_ON(2)) {
      hexagon_controller_->GetPerfInfo();
      hexagon_controller_->PrintLog();
    }
    MACE_CHECK(hexagon_controller_->TeardownGraph(), "hexagon teardown error");
    MACE_CHECK(hexagon_controller_->Finalize(), "hexagon finalize error");
  }
}

MaceStatus MaceEngine::Impl::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  MACE_CHECK_NOTNULL(outputs);
  std::vector<Tensor *> input_tensors;
  std::vector<Tensor *> output_tensors;
  for (auto &input : inputs) {
    Tensor *input_tensor =
        ws_->GetTensor(MakeString("mace_input_node_", input.first, ":0"));
    input_tensor->Resize(input.second.shape());
    {
      Tensor::MappingGuard input_guard(input_tensor);
      float *input_data = input_tensor->mutable_data<float>();
      memcpy(input_data, input.second.data().get(),
             input_tensor->size() * sizeof(float));
    }
    input_tensors.push_back(input_tensor);
  }
  for (auto &output : *outputs) {
    Tensor *output_tensor =
        ws_->GetTensor(MakeString("mace_output_node_", output.first + ":0"));
    output_tensors.push_back(output_tensor);
  }
  if (device_type_ == HEXAGON) {
    MACE_CHECK(input_tensors.size() == 1 && output_tensors.size() == 1,
               "HEXAGON not support multiple inputs and outputs yet.");
    hexagon_controller_->ExecuteGraph(*input_tensors[0], output_tensors[0]);
  } else {
    if (!net_->Run(run_metadata)) {
      LOG(FATAL) << "Net run failed";
    }
  }
  if (device_type_ == OPENCL) {
    OpenCLRuntime::Global()->SaveBuiltCLProgram();
  }
  for (auto &output : *outputs) {
    Tensor *output_tensor =
        ws_->GetTensor(MakeString("mace_output_node_", output.first + ":0"));
    // save output
    if (output_tensor != nullptr && output.second.data() != nullptr) {
      Tensor::MappingGuard output_guard(output_tensor);
      auto shape = output_tensor->shape();
      int64_t output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<int64_t>());
      MACE_CHECK(!shape.empty()) << "Output's shape must greater than 0";
      MACE_CHECK(shape == output.second.shape())
          << "Output shape mispatch: "
          << MakeString<int64_t>(output.second.shape())
          << " != " << MakeString<int64_t>(shape);
      std::memcpy(output.second.data().get(), output_tensor->data<float>(),
                  output_size * sizeof(float));
    } else {
      return MACE_INVALID_ARGS;
    }
  }
  return MACE_SUCCESS;
}

MaceEngine::MaceEngine(const NetDef *net_def,
                       DeviceType device_type,
                       const std::vector<std::string> &input_nodes,
                       const std::vector<std::string> &output_nodes) {
  impl_ = std::unique_ptr<MaceEngine::Impl>(
      new MaceEngine::Impl(net_def, device_type, input_nodes, output_nodes));
}

MaceEngine::~MaceEngine() = default;

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs,
                           RunMetadata *run_metadata) {
  return impl_->Run(inputs, outputs, run_metadata);
}

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs) {
  return impl_->Run(inputs, outputs, nullptr);
}

}  // namespace mace

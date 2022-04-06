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

#ifndef MACE_CORE_RUNTIME_QNN_QNN_WRAPPER_H_
#define MACE_CORE_RUNTIME_QNN_QNN_WRAPPER_H_

#include <dlfcn.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>

#include "mace/core/quantize.h"
#include "mace/core/runtime/runtime.h"
#include "mace/runtimes/qnn/qnn_performance.h"
#include "mace/runtimes/qnn/op_builder.h"
#include "third_party/qnn/include/QnnContext.h"
#include "third_party/qnn/include/QnnProfile.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/runtimes/opencl/core/opencl_helper.h"
#include "mace/runtimes/opencl/opencl_runtime.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {

#ifdef MACE_ENABLE_OPENCL
class QuantizeTransformer {
 public:
  void Init(Runtime *runtime, DataType quantized_type) {
    runtime_ = runtime;
    quantized_type_ = quantized_type;
  }
  MaceStatus Quantize(const Tensor *input, Tensor *output) {
    MACE_LATENCY_LOGGER(1, "Quantize on GPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    // LOG(INFO) << output->scale() << " " << output->zero_point();
    const uint32_t gws = static_cast<uint32_t>(RoundUpDiv4(output->size()));
    MACE_CHECK(runtime_->GetRuntimeType() == RuntimeType::RT_OPENCL);
    OpenclExecutor *executor =
        static_cast<OpenclRuntime *>(runtime_)->GetOpenclExecutor();
    if (quantize_kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name;
      if (quantized_type_ == DT_UINT16) {
        kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_quantize_uint16");
        built_options.emplace("-Dbuffer_quantize_uint16=" + kernel_name);
      }
      else {
        kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_quantize");
        built_options.emplace("-Dbuffer_quantize=" + kernel_name);
      }
      built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
      MACE_RETURN_IF_ERROR(executor->BuildKernel(
          "buffer_transform", kernel_name, built_options, &quantize_kernel_));
    }

    uint32_t idx = 0;
    quantize_kernel_.setArg(idx++, gws);
    quantize_kernel_.setArg(idx++, input->scale());
    quantize_kernel_.setArg(idx++, input->zero_point());
    MACE_CHECK(input->memory_type() == GPU_BUFFER);
    MACE_CHECK(output->memory_type() == GPU_BUFFER);
    quantize_kernel_.setArg(idx++, *(input->memory<cl::Buffer>()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    quantize_kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    quantize_kernel_.setArg(idx++, *(output->memory<cl::Buffer>()));
    
    const uint32_t lws = static_cast<uint32_t>(
        RoundUpDiv4(executor->GetDeviceMaxWorkGroupSize()));
    cl::Event event;
    cl_int error;
    if (executor->IsNonUniformWorkgroupsSupported()) {
      error = executor->command_queue().enqueueNDRangeKernel(
          quantize_kernel_, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws),
          nullptr, &event);
    } else {
      uint32_t roundup_gws = RoundUp(gws, lws);
      error = executor->command_queue().enqueueNDRangeKernel(
          quantize_kernel_, cl::NullRange, cl::NDRange(roundup_gws),
          cl::NDRange(lws), nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Dequantize(const Tensor *input, Tensor *output) {
    MACE_LATENCY_LOGGER(1, "Dequantize on GPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    // LOG(INFO) << output->scale() << " " << output->zero_point();
    const uint32_t gws = static_cast<uint32_t>(RoundUpDiv4(output->size()));
    OpenclExecutor *executor =
        static_cast<OpenclRuntime *>(runtime_)->GetOpenclExecutor();
    if (dequantize_kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name;
      if (quantized_type_ == DT_UINT16) {
        kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_dequantize_uint16");
        built_options.emplace("-Dbuffer_dequantize_uint16=" + kernel_name);
      }
      else {
        kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_dequantize");
        built_options.emplace("-Dbuffer_dequantize=" + kernel_name);
      }
      built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
      MACE_RETURN_IF_ERROR(executor->BuildKernel(
          "buffer_transform", kernel_name, built_options, &dequantize_kernel_));
    }

    uint32_t idx = 0;
    dequantize_kernel_.setArg(idx++, gws);
    dequantize_kernel_.setArg(idx++, input->scale());
    dequantize_kernel_.setArg(idx++, input->zero_point());
    MACE_CHECK(input->memory_type() == GPU_BUFFER);
    MACE_CHECK(output->memory_type() == GPU_BUFFER);
    dequantize_kernel_.setArg(idx++, *(input->memory<cl::Buffer>()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    dequantize_kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    dequantize_kernel_.setArg(idx++, *(output->memory<cl::Buffer>()));

    const uint32_t lws = static_cast<uint32_t>(
        RoundUpDiv4(executor->GetDeviceMaxWorkGroupSize()));
    cl::Event event;
    cl_int error;
    if (executor->IsNonUniformWorkgroupsSupported()) {
      error = executor->command_queue().enqueueNDRangeKernel(
          dequantize_kernel_, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws),
          nullptr, &event);
    } else {
      uint32_t roundup_gws = RoundUp(gws, lws);
      error = executor->command_queue().enqueueNDRangeKernel(
          dequantize_kernel_, cl::NullRange, cl::NDRange(roundup_gws),
          cl::NDRange(lws), nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  Runtime *runtime_;
  DataType quantized_type_;
  cl::Kernel quantize_kernel_;
  cl::Kernel dequantize_kernel_;
};
#else
class QuantizeTransformer {
 public:
  void Init(Runtime *runtime, DataType quantized_type) {
    runtime_ = runtime;
    quantized_type_ = quantized_type;
  }

  MaceStatus Quantize(const Tensor *input, Tensor *output) {
    MACE_LATENCY_LOGGER(1, "Quantize on CPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);

    if (quantized_type_ == DT_UINT16) {
      QuantizeUtil<float, uint16_t> quantize_util_;
      quantize_util_.Init(&runtime_->thread_pool());
      auto output_data = output->mutable_data<uint16_t>();
      auto input_data = input->data<float>();
      quantize_util_.QuantizeWithScaleAndZeropoint(
        input_data, input->size(), input->scale(), input->zero_point(),
        output_data);
    }
    else if (quantized_type_ == DT_UINT8) {
      QuantizeUtil<float, uint8_t> quantize_util_;
      quantize_util_.Init(&runtime_->thread_pool());
      auto output_data = output->mutable_data<uint8_t>();
      auto input_data = input->data<float>();
      quantize_util_.QuantizeWithScaleAndZeropoint(
        input_data, input->size(), input->scale(), input->zero_point(),
        output_data);
    }
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus Dequantize(const Tensor *input, Tensor *output) {
    MACE_LATENCY_LOGGER(1, "Dequantize on CPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    if (quantized_type_ == DT_UINT16) {
      QuantizeUtil<float, uint16_t> quantize_util_;
      quantize_util_.Init(&runtime_->thread_pool());
      auto output_data = output->mutable_data<float>();
      auto input_data = input->data<uint16_t>();
      quantize_util_.Dequantize(input_data, input->size(), input->scale(),
                                input->zero_point(), output_data);
    }
    else if (quantized_type_ == DT_UINT8) {
      QuantizeUtil<float, uint8_t> quantize_util_;
      quantize_util_.Init(&runtime_->thread_pool());
      auto output_data = output->mutable_data<float>();
      auto input_data = input->data<uint8_t>();
      quantize_util_.Dequantize(input_data, input->size(), input->scale(),
                                input->zero_point(), output_data);
    }
    return MaceStatus::MACE_SUCCESS;
  }  
 private:
  Runtime* runtime_;
  DataType quantized_type_;
};
#endif  // MACE_ENABLE_OPENCL

class QnnWrapper {
 public:
  QnnWrapper(Runtime *runtime);

  std::string GetVersion();
  bool Init(const NetDef &net_def,
            unsigned const char *model_data,
            const index_t model_data_size,
            const AcceleratorCachePolicy cache_policy,
            const std::string &cache_binary_file,
            const std::string &cache_storage_file,
            HexagonPerformanceType perf_type_);
  bool InitOnline(const NetDef &net_def,
                  unsigned const char *model_data,
                  const index_t model_data_size);
  bool InitWithOfflineCache(const NetDef &net_def,
                            const std::string &cache_binary_file);
  bool CacheStore(const NetDef &net_def,
                  const std::string &cache_storage_file);

  bool SetPerformance(const HexagonPerformanceType type);
  bool SetPerformance(const QnnGraphState state,
                      const HexagonPerformanceType type);
  bool Run(const std::map<std::string, Tensor*> &input_tensors,
           std::map<std::string, Tensor*> *output_tensors);
  bool Destroy();

  void PrepareBackend();
  StatusCode getQnnFunctionPointers(std::string backendPath);
  ~QnnWrapper() {
    if (backend_handle_) {
      dlclose(backend_handle_);
      backend_handle_ = nullptr;
    }
  }
 private:
  void GetPerfInfo();
  void CollectPerfInfo();
  void CollectOpInfo(const NetDef &net_def);
  void GetEvent(QnnProfile_EventId_t event, 
                bool collect_op_infos);
  void GetSubEvents(QnnProfile_EventId_t event);
  QnnGraphState graph_state_;
  HexagonPerformanceType perf_type_;
  std::unique_ptr<QnnPerformance> perf_;
  Qnn_ContextHandle_t ctx_;
  Qnn_GraphHandle_t graph_;
  GraphBuilder graph_builder_;
  Qnn_ProfileHandle_t profile_;
  QnnProfile_Level_t profile_level_;
  int num_inputs_;
  int num_outputs_;
  std::vector<QnnInOutInfo> input_info_;
  std::vector<QnnInOutInfo> output_info_;
  std::vector<Qnn_Tensor_t> input_tensors_;
  std::vector<Qnn_Tensor_t> output_tensors_;
  struct ProfileInfo{
    std::vector<std::pair<std::string, std::string>> op_infos;
    std::vector<std::vector<uint64_t>> op_cycles;
    std::vector<std::vector<uint32_t>> output_shapes;
    bool is_warmup = true;
    unsigned int qnn_time;
    unsigned int npu_time;
    uint64_t npu_cycle;
  };
  ProfileInfo profile_info_;
  QuantizeTransformer *transformer_;
  Runtime *runtime_;
  DataType quantized_type_;
  QnnFunctionPointers qnn_function_pointers_;
  void* backend_handle_ = nullptr;
  MACE_DISABLE_COPY_AND_ASSIGN(QnnWrapper);
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_QNN_QNN_WRAPPER_H_

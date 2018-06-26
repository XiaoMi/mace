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

#include "src/main/cpp/image_classify.h"

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include "src/main/cpp/include/mace/public/mace.h"
#include "src/main/cpp/include/mace/public/mace_runtime.h"
#include "src/main/cpp/include/mace/public/mace_engine_factory.h"

namespace {

struct ModelInfo {
  std::string input_name;
  std::string output_name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
};

struct MaceContext {
  std::shared_ptr<mace::MaceEngine> engine;
  std::shared_ptr<mace::KVStorageFactory> storage_factory;
  std::string model_name;
  mace::DeviceType device_type = mace::DeviceType::CPU;
  std::map<std::string, ModelInfo> model_infos = {
      {"mobilenet_v1", {"input", "MobilenetV1/Predictions/Reshape_1",
                            {1, 224, 224, 3}, {1, 1001}}},
      {"mobilenet_v2", {"input", "MobilenetV2/Predictions/Reshape_1",
                            {1, 224, 224, 3}, {1, 1001}}}
  };
};

mace::DeviceType ParseDeviceType(const std::string &device) {
  if (device.compare("CPU") == 0) {
    return mace::DeviceType::CPU;
  } else if (device.compare("GPU") == 0) {
    return mace::DeviceType::GPU;
  } else if (device.compare("HEXAGON") == 0) {
    return mace::DeviceType::HEXAGON;
  } else {
    return mace::DeviceType::CPU;
  }
}

MaceContext& GetMaceContext() {
  // stay for the app's life time, only initialize once
  static auto *mace_context = new MaceContext;

  return *mace_context;
}

}  // namespace

JNIEXPORT jint JNICALL Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetSetAttrs(
    JNIEnv *env, jclass thisObj, jint omp_num_threads, jint cpu_affinity_policy,
    jint gpu_perf_hint, jint gpu_priority_hint, jstring kernel_path) {
  MaceContext &mace_context = GetMaceContext();

  mace::MaceStatus status;
      // openmp
  status = mace::SetOpenMPThreadPolicy(
      omp_num_threads,
      static_cast<mace::CPUAffinityPolicy>(cpu_affinity_policy));

  __android_log_print(ANDROID_LOG_ERROR,
                      "image_classify attrs",
                      "openmp result: %d, threads: %d, cpu: %d",
                      status, omp_num_threads, cpu_affinity_policy);

  //  gpu
  mace::SetGPUHints(
      static_cast<mace::GPUPerfHint>(gpu_perf_hint),
      static_cast<mace::GPUPriorityHint>(gpu_priority_hint));

  __android_log_print(ANDROID_LOG_ERROR,
                      "image_classify attrs",
                      "gpu perf: %d, priority: %d",
                      gpu_perf_hint, gpu_priority_hint);

  //  opencl cache
  const char *kernel_path_ptr = env->GetStringUTFChars(kernel_path, nullptr);
  if (kernel_path_ptr == nullptr) return JNI_ERR;
  const std::string kernel_file_path(kernel_path_ptr);
  mace_context.storage_factory.reset(
      new mace::FileStorageFactory(kernel_file_path));
  mace::SetKVStorageFactory(mace_context.storage_factory);
  env->ReleaseStringUTFChars(kernel_path, kernel_path_ptr);

  return JNI_OK;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateEngine(
    JNIEnv *env, jclass thisObj, jstring model_name_str, jstring device) {
  MaceContext &mace_context = GetMaceContext();
  //  parse model name
  const char *model_name_ptr = env->GetStringUTFChars(model_name_str, nullptr);
  if (model_name_ptr == nullptr) return JNI_ERR;
  mace_context.model_name.assign(model_name_ptr);
  env->ReleaseStringUTFChars(model_name_str, model_name_ptr);

  //  load model input and output name
  auto model_info_iter =
      mace_context.model_infos.find(mace_context.model_name);
  if (model_info_iter == mace_context.model_infos.end()) {
    __android_log_print(ANDROID_LOG_ERROR,
                        "image_classify",
                        "Invalid model name: %s",
                        mace_context.model_name.c_str());
    return JNI_ERR;
  }
  std::vector<std::string> input_names = {model_info_iter->second.input_name};
  std::vector<std::string> output_names = {model_info_iter->second.output_name};

  // get device
  const char *device_ptr = env->GetStringUTFChars(device, nullptr);
  if (device_ptr == nullptr) return JNI_ERR;
  mace_context.device_type = ParseDeviceType(device_ptr);
  env->ReleaseStringUTFChars(device, device_ptr);

  __android_log_print(ANDROID_LOG_ERROR,
                      "image_classify attrs",
                      "device: %d",
                      mace_context.device_type);

  mace::MaceStatus create_engine_status =
      CreateMaceEngineFromCode(mace_context.model_name,
                               std::string(),
                               input_names,
                               output_names,
                               mace_context.device_type,
                               &mace_context.engine);

  __android_log_print(ANDROID_LOG_ERROR,
                      "image_classify attrs",
                      "create result: %d",
                      create_engine_status);

  return create_engine_status == mace::MaceStatus::MACE_SUCCESS ?
         JNI_OK : JNI_ERR;
}

JNIEXPORT jfloatArray JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetClassify(
    JNIEnv *env, jclass thisObj, jfloatArray input_data) {
  MaceContext &mace_context = GetMaceContext();
  //  prepare input and output
  auto model_info_iter =
      mace_context.model_infos.find(mace_context.model_name);
  if (model_info_iter == mace_context.model_infos.end()) {
    __android_log_print(ANDROID_LOG_ERROR,
                        "image_classify",
                        "Invalid model name: %s",
                        mace_context.model_name.c_str());
    return nullptr;
  }
  const ModelInfo &model_info = model_info_iter->second;
  const std::string &input_name = model_info.input_name;
  const std::string &output_name = model_info.output_name;
  const std::vector<int64_t> &input_shape = model_info.input_shape;
  const std::vector<int64_t> &output_shape = model_info.output_shape;
  const int64_t input_size =
      std::accumulate(input_shape.begin(), input_shape.end(), 1,
                      std::multiplies<int64_t>());
  const int64_t output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), 1,
                      std::multiplies<int64_t>());

  //  load input
  jfloat *input_data_ptr = env->GetFloatArrayElements(input_data, nullptr);
  if (input_data_ptr == nullptr) return nullptr;
  jsize length = env->GetArrayLength(input_data);
  if (length != input_size) return nullptr;

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  // construct input
  auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                          std::default_delete<float[]>());
  std::copy_n(input_data_ptr, input_size, buffer_in.get());
  env->ReleaseFloatArrayElements(input_data, input_data_ptr, 0);
  inputs[input_name] = mace::MaceTensor(input_shape, buffer_in);

  // construct output
  auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                           std::default_delete<float[]>());
  outputs[output_name] = mace::MaceTensor(output_shape, buffer_out);

  // run model
  mace_context.engine->Run(inputs, &outputs);

  // transform output
  jfloatArray jOutputData = env->NewFloatArray(output_size);  // allocate
  if (jOutputData == nullptr) return nullptr;
  env->SetFloatArrayRegion(jOutputData, 0, output_size,
                           outputs[output_name].data().get());  // copy

  return jOutputData;
}

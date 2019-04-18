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

/**
 * Usage:
 * throughput_test \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --cpu_model_data_file=cpu_model_data.data \
 *          --gpu_model_data_file=gpu_model_data.data \
 *          --dsp_model_data_file=dsp_model_data.data \
 *          --run_seconds=10
 */
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>  // NOLINT(build/c++11)

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/utils/logging.h"
#include "mace/core/types.h"

namespace mace {

#ifdef MACE_CPU_MODEL_TAG
namespace MACE_CPU_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_CPU_MODEL_TAG
#endif

#ifdef MACE_GPU_MODEL_TAG
namespace MACE_GPU_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_GPU_MODEL_TAG
#endif

#ifdef MACE_DSP_MODEL_TAG
namespace MACE_DSP_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_DSP_MODEL_TAG
#endif

namespace benchmark {

void Split(const std::string &str,
           char delims,
           std::vector<std::string> *result) {
  MACE_CHECK_NOTNULL(result);
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result->push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

void SplitAndParseToInts(const std::string &str,
                         char delims,
                         std::vector<int64_t> *result) {
  MACE_CHECK_NOTNULL(result);
  std::string tmp = str;
  while (!tmp.empty()) {
    index_t dim = atoi(tmp.data());
    result->push_back(dim);
    size_t next_offset = tmp.find(delims);
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    index_t dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!::isalnum(res[i])) res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("GPU") == 0) {
    return DeviceType::GPU;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else {
    return DeviceType::CPU;
  }
}

DEFINE_string(input_node, "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(output_node, "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(input_shape, "1,224,224,3", "input shape, separated by comma");
DEFINE_string(output_shape, "1,224,224,2", "output shape, separated by comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_string(cpu_model_data_file, "", "cpu model data file name");
DEFINE_string(gpu_model_data_file, "", "gpu model data file name");
DEFINE_string(dsp_model_data_file, "", "dsp model data file name");
DEFINE_int32(run_seconds, 10, "run seconds");

int Main(int argc, char **argv) {
  std::string usage = "model throughput test\nusage: " + std::string(argv[0])
      + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "mace version: " << MaceVersion();
#ifdef MACE_CPU_MODEL_TAG
  LOG(INFO) << "cpu model checksum: "
            << mace::MACE_CPU_MODEL_TAG::ModelChecksum();
#endif
#ifdef MACE_GPU_MODEL_TAG
  LOG(INFO) << "gpu model checksum: "
            << mace::MACE_GPU_MODEL_TAG::ModelChecksum();
#endif
#ifdef MACE_DSP_MODEL_TAG
  LOG(INFO) << "dsp model checksum: "
            << mace::MACE_DSP_MODEL_TAG::ModelChecksum();
#endif
  LOG(INFO) << "Input node: [" << FLAGS_input_node<< "]";
  LOG(INFO) << "input_shape: " << FLAGS_input_shape;
  LOG(INFO) << "Output node: [" << FLAGS_output_node<< "]";
  LOG(INFO) << "output_shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "cpu_model_data_file: " << FLAGS_cpu_model_data_file;
  LOG(INFO) << "gpu_model_data_file: " << FLAGS_gpu_model_data_file;
  LOG(INFO) << "dsp_model_data_file: " << FLAGS_dsp_model_data_file;
  LOG(INFO) << "run_seconds: " << FLAGS_run_seconds;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> input_shapes;
  std::vector<std::string> output_shapes;
  Split(FLAGS_input_node, ',', &input_names);
  Split(FLAGS_output_node, ',', &output_names);
  Split(FLAGS_input_shape, ':', &input_shapes);
  Split(FLAGS_output_shape, ':', &output_shapes);

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int64_t>> input_shape_vec(input_count);
  std::vector<std::vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> cpu_outputs;
  std::map<std::string, mace::MaceTensor> gpu_outputs;
  std::map<std::string, mace::MaceTensor> dsp_outputs;
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    int64_t input_size =
        std::accumulate(input_shape_vec[i].begin(), input_shape_vec[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    // load input
    std::ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]),
                          std::ios::in | std::ios::binary);
    if (in_file.is_open()) {
      in_file.read(reinterpret_cast<char *>(buffer_in.get()),
                   input_size * sizeof(float));
      in_file.close();
    } else {
      LOG(FATAL) << "Open input file failed";
    }
    inputs[input_names[i]] = mace::MaceTensor(input_shape_vec[i], buffer_in);
  }

  for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shape_vec[i].begin(),
                        output_shape_vec[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    cpu_outputs[output_names[i]] = mace::MaceTensor(output_shape_vec[i],
                                                    buffer_out);
    gpu_outputs[output_names[i]] = mace::MaceTensor(output_shape_vec[i],
                                                    buffer_out);
    dsp_outputs[output_names[i]] = mace::MaceTensor(output_shape_vec[i],
                                                    buffer_out);
  }

#if defined(MACE_CPU_MODEL_TAG) || \
    defined(MACE_GPU_MODEL_TAG) || \
    defined(MACE_DSP_MODEL_TAG)
  int64_t t0, t1, init_micros;
#endif

#ifdef MACE_CPU_MODEL_TAG
  /* --------------------- CPU init ----------------------- */
  LOG(INFO) << "Load & init cpu model and warm up";
  const unsigned char *cpu_model_data =
      mace::MACE_CPU_MODEL_TAG::LoadModelData(
      FLAGS_cpu_model_data_file.c_str());
  NetDef cpu_net_def = mace::MACE_CPU_MODEL_TAG::CreateNet(cpu_model_data);

  mace::MaceEngine cpu_engine(&cpu_net_def, DeviceType::CPU, input_names,
                              output_names);

  LOG(INFO) << "CPU Warm up run";
  t0 = NowMicros();
  cpu_engine.Run(inputs, &cpu_outputs);
  t1 = NowMicros();
  LOG(INFO) << "CPU 1st warm up run latency: " << t1 - t0 << " us";
#endif

#ifdef MACE_GPU_MODEL_TAG
  /* --------------------- GPU init ----------------------- */
  LOG(INFO) << "Load & init gpu model and warm up";
  const unsigned char *gpu_model_data =
      mace::MACE_GPU_MODEL_TAG::LoadModelData(
      FLAGS_gpu_model_data_file.c_str());
  NetDef gpu_net_def = mace::MACE_GPU_MODEL_TAG::CreateNet(gpu_model_data);

  mace::MaceEngine gpu_engine(&gpu_net_def, DeviceType::GPU, input_names,
                              output_names);
  mace::MACE_GPU_MODEL_TAG::UnloadModelData(gpu_model_data);

  LOG(INFO) << "GPU Warm up run";
  t0 = NowMicros();
  gpu_engine.Run(inputs, &gpu_outputs);
  t1 = NowMicros();
  LOG(INFO) << "GPU 1st warm up run latency: " << t1 - t0 << " us";
#endif

#ifdef MACE_DSP_MODEL_TAG
  /* --------------------- DSP init ----------------------- */
  LOG(INFO) << "Load & init dsp model and warm up";
  const unsigned char *dsp_model_data =
      mace::MACE_DSP_MODEL_TAG::LoadModelData(
      FLAGS_dsp_model_data_file.c_str());
  NetDef dsp_net_def = mace::MACE_DSP_MODEL_TAG::CreateNet(dsp_model_data);

  mace::MaceEngine dsp_engine(&dsp_net_def, DeviceType::HEXAGON, input_names,
                              output_names);
  mace::MACE_DSP_MODEL_TAG::UnloadModelData(dsp_model_data);

  LOG(INFO) << "DSP Warm up run";
  t0 = NowMicros();
  dsp_engine.Run(inputs, &dsp_outputs);
  t1 = NowMicros();
  LOG(INFO) << "DSP 1st warm up run latency: " << t1 - t0 << " us";
#endif

#if defined(MACE_CPU_MODEL_TAG) || \
    defined(MACE_GPU_MODEL_TAG) || \
    defined(MACE_DSP_MODEL_TAG)
  double cpu_throughput = 0;
  double gpu_throughput = 0;
  double dsp_throughput = 0;
  int64_t run_micros = FLAGS_run_seconds * 1000000;
#endif

#ifdef MACE_CPU_MODEL_TAG
  std::thread cpu_thread([&]() {
    int64_t frames = 0;
    int64_t micros = 0;
    int64_t start = NowMicros();
    for (; micros < run_micros; ++frames) {
      cpu_engine.Run(inputs, &cpu_outputs);
      int64_t end = NowMicros();
      micros = end - start;
    }
    cpu_throughput = frames * 1000000.0 / micros;
  });
#endif

#ifdef MACE_GPU_MODEL_TAG
  std::thread gpu_thread([&]() {
    int64_t frames = 0;
    int64_t micros = 0;
    int64_t start = NowMicros();
    for (; micros < run_micros; ++frames) {
      gpu_engine.Run(inputs, &gpu_outputs);
      int64_t end = NowMicros();
      micros = end - start;
    }
    gpu_throughput = frames * 1000000.0 / micros;
  });
#endif

#ifdef MACE_DSP_MODEL_TAG
  std::thread dsp_thread([&]() {
    int64_t frames = 0;
    int64_t micros = 0;
    int64_t start = NowMicros();
    for (; micros < run_micros; ++frames) {
      dsp_engine.Run(inputs, &dsp_outputs);
      int64_t end = NowMicros();
      micros = end - start;
    }
    dsp_throughput = frames * 1000000.0 / micros;
  });
#endif

  double total_throughput = 0;

#ifdef MACE_CPU_MODEL_TAG
  cpu_thread.join();
  LOG(INFO) << "CPU throughput: " << cpu_throughput << " f/s";
  total_throughput += cpu_throughput;
#endif
#ifdef MACE_GPU_MODEL_TAG
  gpu_thread.join();
  LOG(INFO) << "GPU throughput: " << gpu_throughput << " f/s";
  total_throughput += gpu_throughput;
#endif
#ifdef MACE_DSP_MODEL_TAG
  dsp_thread.join();
  LOG(INFO) << "DSP throughput: " << dsp_throughput << " f/s";
  total_throughput += dsp_throughput;
#endif

  LOG(INFO) << "Total throughput: " << total_throughput << " f/s";

  return 0;
}

}  // namespace benchmark
}  // namespace mace

int main(int argc, char **argv) { mace::benchmark::Main(argc, argv); }

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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
#include <malloc.h>
#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

using namespace std;
using namespace mace;

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

}  // namespace mace

void ParseShape(const string &str, vector<int64_t> *shape) {
  string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

DeviceType ParseDeviceType(const string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("NEON") == 0) {
    return DeviceType::NEON;
  } else if (device_str.compare("OPENCL") == 0) {
    return DeviceType::OPENCL;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else {
    return DeviceType::CPU;
  }
}

DEFINE_string(input_shape, "1,224,224,3", "input shape, separated by comma");
DEFINE_string(output_shape, "1,224,224,2", "output shape, separated by comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_string(cpu_model_data_file, "", "cpu model data file name");
DEFINE_string(gpu_model_data_file, "", "gpu model data file name");
DEFINE_string(dsp_model_data_file, "", "dsp model data file name");
DEFINE_int32(run_seconds, 10, "run seconds");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "mace git version: " << MaceGitVersion();
#ifdef MACE_CPU_MODEL_TAG
  LOG(INFO) << "cpu model checksum: " << mace::MACE_CPU_MODEL_TAG::ModelChecksum();
#endif
#ifdef MACE_GPU_MODEL_TAG
  LOG(INFO) << "gpu model checksum: " << mace::MACE_GPU_MODEL_TAG::ModelChecksum();
#endif
#ifdef MACE_DSP_MODEL_TAG
  LOG(INFO) << "dsp model checksum: " << mace::MACE_DSP_MODEL_TAG::ModelChecksum();
#endif
  LOG(INFO) << "input_shape: " << FLAGS_input_shape;
  LOG(INFO) << "output_shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "cpu_model_data_file: " << FLAGS_cpu_model_data_file;
  LOG(INFO) << "gpu_model_data_file: " << FLAGS_gpu_model_data_file;
  LOG(INFO) << "dsp_model_data_file: " << FLAGS_dsp_model_data_file;
  LOG(INFO) << "run_seconds: " << FLAGS_run_seconds;

  vector<int64_t> input_shape_vec;
  vector<int64_t> output_shape_vec;
  ParseShape(FLAGS_input_shape, &input_shape_vec);
  ParseShape(FLAGS_output_shape, &output_shape_vec);

  int64_t input_size =
      std::accumulate(input_shape_vec.begin(), input_shape_vec.end(), 1,
                      std::multiplies<int64_t>());
  int64_t output_size =
      std::accumulate(output_shape_vec.begin(), output_shape_vec.end(), 1,
                      std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> cpu_output_data(new float[output_size]);
  std::unique_ptr<float[]> gpu_output_data(new float[output_size]);
  std::unique_ptr<float[]> dsp_output_data(new float[output_size]);

  // load input
  ifstream in_file(FLAGS_input_file, ios::in | ios::binary);
  if (in_file.is_open()) {
    in_file.read(reinterpret_cast<char *>(input_data.get()),
                 input_size * sizeof(float));
    in_file.close();
  } else {
    LOG(INFO) << "Open input file failed";
    return -1;
  }

  int64_t t0, t1, init_micros;
#ifdef MACE_CPU_MODEL_TAG
  /* --------------------- CPU init ----------------------- */
  LOG(INFO) << "Load & init cpu model and warm up";
  const unsigned char *cpu_model_data =
      mace::MACE_CPU_MODEL_TAG::LoadModelData(FLAGS_cpu_model_data_file.c_str());
  NetDef cpu_net_def = mace::MACE_CPU_MODEL_TAG::CreateNet(cpu_model_data);

  mace::MaceEngine cpu_engine(&cpu_net_def, DeviceType::CPU);

  LOG(INFO) << "CPU Warm up run";
  t0 = NowMicros();
  cpu_engine.Run(input_data.get(), input_shape_vec, cpu_output_data.get());
  t1 = NowMicros();
  LOG(INFO) << "CPU 1st warm up run latency: " << t1 - t0 << " us";
#endif

#ifdef MACE_GPU_MODEL_TAG
  /* --------------------- GPU init ----------------------- */
  LOG(INFO) << "Load & init gpu model and warm up";
  const unsigned char *gpu_model_data =
      mace::MACE_GPU_MODEL_TAG::LoadModelData(FLAGS_gpu_model_data_file.c_str());
  NetDef gpu_net_def = mace::MACE_GPU_MODEL_TAG::CreateNet(gpu_model_data);

  mace::MaceEngine gpu_engine(&gpu_net_def, DeviceType::OPENCL);
  mace::MACE_GPU_MODEL_TAG::UnloadModelData(gpu_model_data);

  LOG(INFO) << "GPU Warm up run";
  t0 = NowMicros();
  gpu_engine.Run(input_data.get(), input_shape_vec, gpu_output_data.get());
  t1 = NowMicros();
  LOG(INFO) << "GPU 1st warm up run latency: " << t1 - t0 << " us";
#endif

#ifdef MACE_DSP_MODEL_TAG
  /* --------------------- DSP init ----------------------- */
  LOG(INFO) << "Load & init dsp model and warm up";
  const unsigned char *dsp_model_data =
      mace::MACE_DSP_MODEL_TAG::LoadModelData(FLAGS_gpu_model_data_file.c_str());
  NetDef dsp_net_def = mace::MACE_DSP_MODEL_TAG::CreateNet(dsp_model_data);

  mace::MaceEngine dsp_engine(&dsp_net_def, DeviceType::HEXAGON);
  mace::MACE_DSP_MODEL_TAG::UnloadModelData(dsp_model_data);

  LOG(INFO) << "DSP Warm up run";
  t0 = NowMicros();
  gpu_engine.Run(input_data.get(), input_shape_vec, dsp_output_data.get());
  t1 = NowMicros();
  LOG(INFO) << "DSP 1st warm up run latency: " << t1 - t0 << " us";
#endif

  double cpu_throughput = 0;
  double gpu_throughput = 0;
  double dsp_throughput = 0;
  int64_t run_micros = FLAGS_run_seconds * 1000000;

#ifdef MACE_CPU_MODEL_TAG
  std::thread cpu_thread([&]() {
    int64_t frames = 0;
    int64_t micros = 0;
    int64_t start = NowMicros();
    for (; micros < run_micros; ++frames) {
      cpu_engine.Run(input_data.get(), input_shape_vec, cpu_output_data.get());
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
      gpu_engine.Run(input_data.get(), input_shape_vec, gpu_output_data.get());
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
      dsp_engine.Run(input_data.get(), input_shape_vec, dsp_output_data.get());
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
}

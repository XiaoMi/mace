//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

/**
 * Usage:
 * mace_run --model=mobi_mace.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mace.out  \
 *          --model_data_file=model_data.data \
 *          --device=OPENCL
 */
#include <malloc.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

// #include "mace/codegen/models/${MACE_MODEL_TAG}/${MACE_MODEL_TAG}.h" instead
namespace mace {
namespace MACE_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_MODEL_TAG
}  // namespace mace

namespace mace {
namespace examples {

namespace str_util {

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result.push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}

}  // namespace str_util

void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
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
    if (!isalnum(res[i])) res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
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

void WriteOpenCLPlatformInfo(const std::string &output_dir) {
  std::string platform_info = OpenCLRuntime::Global()->platform_info();
  const std::string cl_platform_info_file_name = output_dir
      + "/mace_cl_platform_info.txt";

  std::ofstream ofs(cl_platform_info_file_name);
  if (ofs.is_open()) {
    ofs << platform_info;
    ofs.close();
  } else {
    LOG(WARNING) << "Write opencl platform info failed.";
  }
}

struct mallinfo LogMallinfoChange(struct mallinfo prev) {
  struct mallinfo curr = mallinfo();
  if (prev.arena != curr.arena) {
    LOG(INFO) << "Non-mmapped space allocated (bytes): " << curr.arena
              << ", diff: " << ((int64_t)curr.arena - (int64_t)prev.arena);
  }
  if (prev.ordblks != curr.ordblks) {
    LOG(INFO) << "Number of free chunks: " << curr.ordblks
              << ", diff: " << ((int64_t)curr.ordblks - (int64_t)prev.ordblks);
  }
  if (prev.smblks != curr.smblks) {
    LOG(INFO) << "Number of free fastbin blocks: " << curr.smblks
              << ", diff: " << ((int64_t)curr.smblks - (int64_t)prev.smblks);
  }
  if (prev.hblks != curr.hblks) {
    LOG(INFO) << "Number of mmapped regions: " << curr.hblks
              << ", diff: " << ((int64_t)curr.hblks - (int64_t)prev.hblks);
  }
  if (prev.hblkhd != curr.hblkhd) {
    LOG(INFO) << "Space allocated in mmapped regions (bytes): " << curr.hblkhd
              << ", diff: " << ((int64_t)curr.hblkhd - (int64_t)prev.hblkhd);
  }
  if (prev.usmblks != curr.usmblks) {
    LOG(INFO) << "Maximum total allocated space (bytes): " << curr.usmblks
              << ", diff: " << ((int64_t)curr.usmblks - (int64_t)prev.usmblks);
  }
  if (prev.fsmblks != curr.fsmblks) {
    LOG(INFO) << "Space in freed fastbin blocks (bytes): " << curr.fsmblks
              << ", diff: " << ((int64_t)curr.fsmblks - (int64_t)prev.fsmblks);
  }
  if (prev.uordblks != curr.uordblks) {
    LOG(INFO) << "Total allocated space (bytes): " << curr.uordblks
              << ", diff: "
              << ((int64_t)curr.uordblks - (int64_t)prev.uordblks);
  }
  if (prev.fordblks != curr.fordblks) {
    LOG(INFO) << "Total free space (bytes): " << curr.fordblks << ", diff: "
              << ((int64_t)curr.fordblks - (int64_t)prev.fordblks);
  }
  if (prev.keepcost != curr.keepcost) {
    LOG(INFO) << "Top-most, releasable space (bytes): " << curr.keepcost
              << ", diff: "
              << ((int64_t)curr.keepcost - (int64_t)prev.keepcost);
  }
  return curr;
}

DEFINE_string(input_node,
              "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "1,224,224,3:1,1,1,10",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "1,224,224,2:1,1,1,10",
              "output shapes, separated by colon and comma");
DEFINE_string(input_file,
              "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "",
              "output file name | output file prefix for multiple outputs");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_string(device, "OPENCL", "CPU/NEON/OPENCL/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 2, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 1, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, 8, "num of openmp threads");
DEFINE_int32(cpu_power_option,
             0,
             "0:DEFAULT/1:HIGH_PERFORMANCE/2:BATTERY_SAVE");

bool RunModel(const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes) {
  // load model
  int64_t t0 = NowMicros();
  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);
  int64_t t1 = NowMicros();
  double create_net_millis = (t1 - t0) / 1000.0;
  LOG(INFO) << "CreateNetDef latency: " << create_net_millis << " ms";

  DeviceType device_type = ParseDeviceType(FLAGS_device);
  LOG(INFO) << "Runing with device type: " << device_type;

  // config runtime
  mace::ConfigOmpThreads(FLAGS_omp_num_threads);
  mace::ConfigCPUPowerOption(
      static_cast<CPUPowerOption>(FLAGS_cpu_power_option));
  if (device_type == DeviceType::OPENCL) {
    mace::ConfigOpenCLRuntime(
        static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
        static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
  }

  const char *kernel_path = getenv("MACE_CL_PROGRAM_PATH");
  const std::string kernel_file_path =
      std::string(kernel_path == nullptr ?
                  "/data/local/tmp/mace_run/cl_program" : kernel_path);

  // Init model
  LOG(INFO) << "Run init";
  std::shared_ptr<KVStorageFactory> storage_factory(
      new FileStorageFactory(kernel_file_path));
  ConfigKVStorageFactory(storage_factory);
  mace::MaceEngine engine(&net_def, device_type, input_names, output_names);
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }
  int64_t t2 = NowMicros();
  double mace_engine_ctor_millis = (t2 - t1) / 1000.0;
  double init_millis = (t2 - t0) / 1000.0;
  LOG(INFO) << "MaceEngine constructor latency: "
            << mace_engine_ctor_millis << " ms";
  LOG(INFO) << "Total init latency: " << init_millis << " ms";

  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    int64_t input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
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
      LOG(INFO) << "Open input file failed";
      return -1;
    }
    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
  }

  for (size_t i = 0; i < output_count; ++i) {
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
  }

  LOG(INFO) << "Warm up run";
  int64_t t3 = NowMicros();
  engine.Run(inputs, &outputs);
  int64_t t4 = NowMicros();
  double warmup_millis = (t4 - t3) / 1000.0;
  LOG(INFO) << "1st warm up run latency: " << warmup_millis << " ms";

  double model_run_millis = -1;
  if (FLAGS_round > 0) {
    LOG(INFO) << "Run model";
    int64_t t0 = NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < FLAGS_round; ++i) {
      engine.Run(inputs, &outputs);
      if (FLAGS_malloc_check_cycle >= 1 && i % FLAGS_malloc_check_cycle == 0) {
        LOG(INFO) << "=== check malloc info change #" << i << " ===";
        prev = LogMallinfoChange(prev);
      }
    }
    int64_t t1 = NowMicros();
    model_run_millis = (t1 - t0) / 1000.0 / FLAGS_round;
    LOG(INFO) << "Average latency: " << model_run_millis << " ms";
  }

  // Metrics reporting tools depends on the format, keep in consistent
  printf("================================================================\n");
  printf("      create_net engine_ctor        init      warmup     run_avg\n");
  printf("================================================================\n");
  printf("time %11.3f %11.3f %11.3f %11.3f %11.3f\n", create_net_millis,
         mace_engine_ctor_millis, init_millis, warmup_millis, model_run_millis);

  WriteOpenCLPlatformInfo(kernel_file_path);

  for (size_t i = 0; i < output_count; ++i) {
    std::string output_name =
        FLAGS_output_file + "_" + FormatName(output_names[i]);
    std::ofstream out_file(output_name, std::ios::binary);
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    out_file.write(
        reinterpret_cast<char *>(outputs[output_names[i]].data().get()),
        output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file " << output_name << " with size "
              << output_size << " done.";
  }

  return true;
}

int Main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "model checksum: " << mace::MACE_MODEL_TAG::ModelChecksum();
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "output node: " << FLAGS_output_node;
  LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "device: " << FLAGS_device;
  LOG(INFO) << "round: " << FLAGS_round;
  LOG(INFO) << "restart_round: " << FLAGS_restart_round;
  LOG(INFO) << "gpu_perf_hint: " << FLAGS_gpu_perf_hint;
  LOG(INFO) << "gpu_priority_hint: " << FLAGS_gpu_priority_hint;
  LOG(INFO) << "omp_num_threads: " << FLAGS_omp_num_threads;
  LOG(INFO) << "cpu_power_option: " << FLAGS_cpu_power_option;

  std::vector<std::string> input_names = str_util::Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names =
      str_util::Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes =
      str_util::Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes =
      str_util::Split(FLAGS_output_shape, ':');

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

  bool ret;
#pragma omp parallel for
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    VLOG(0) << "restart round " << i;
    ret =
        RunModel(input_names, input_shape_vec, output_names, output_shape_vec);
  }
  if (ret) {
    return 0;
  } else {
    return -1;
  }
}

}  // namespace examples
}  // namespace mace

int main(int argc, char **argv) { mace::examples::Main(argc, argv); }

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//


#include <sys/time.h>

#include <cstdlib>
#include <fstream>
#include <numeric>
#include <thread>  // NOLINT(build/c++11)

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/logging.h"
#include "mace/benchmark/stat_summarizer.h"

namespace mace {
namespace MACE_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_MODEL_TAG
}  // namespace mace

namespace mace {
namespace benchmark {
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

bool SplitAndParseToInts(const std::string &str,
                         char delims,
                         std::vector<int64_t> *result) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int64_t dim = atoi(tmp.data());
    result->push_back(dim);
    size_t next_offset = tmp.find(delims);
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return true;
}

}  //  namespace str_util

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
    if (!::isalnum(res[i])) res[i] = '_';
  }
  return res;
}

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
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

bool RunInference(MaceEngine *engine,
                  const std::map<std::string, mace::MaceTensor> &input_infos,
                  std::map<std::string, mace::MaceTensor> *output_infos,
                  StatSummarizer *summarizer,
                  int64_t *inference_time_us) {
  MACE_CHECK_NOTNULL(output_infos);
  RunMetadata run_metadata;
  RunMetadata *run_metadata_ptr = nullptr;
  if (summarizer) {
    run_metadata_ptr = &run_metadata;
  }

  const int64_t start_time = NowMicros();
  mace::MaceStatus s = engine->Run(input_infos, output_infos, run_metadata_ptr);
  const int64_t end_time = NowMicros();

  if (s != mace::MaceStatus::MACE_SUCCESS) {
    LOG(ERROR) << "Error during inference.";
    return false;
  }
  *inference_time_us = end_time - start_time;

  if (summarizer != nullptr) {
    summarizer->ProcessMetadata(run_metadata);
  }

  return true;
}

bool Run(MaceEngine *engine,
         const std::map<std::string, mace::MaceTensor> &input_infos,
         std::map<std::string, mace::MaceTensor> *output_infos,
         StatSummarizer *summarizer,
         int num_runs,
         double max_time_sec,
         int64_t sleep_sec,
         int64_t *total_time_us,
         int64_t *actual_num_runs) {
  MACE_CHECK_NOTNULL(output_infos);
  *total_time_us = 0;

  LOG(INFO) << "Running benchmark for max " << num_runs << " iterators, max "
            << max_time_sec << " seconds "
            << (summarizer != nullptr ? "with " : "without ")
            << "detailed stat logging, with " << sleep_sec
            << "s sleep between inferences";

  Stat<int64_t> stat;

  bool util_max_time = (num_runs <= 0);
  for (int i = 0; util_max_time || i < num_runs; ++i) {
    int64_t inference_time_us = 0;
    bool s = RunInference(engine, input_infos, output_infos,
                          summarizer, &inference_time_us);
    stat.UpdateStat(inference_time_us);
    (*total_time_us) += inference_time_us;
    ++(*actual_num_runs);

    if (max_time_sec > 0 && (*total_time_us / 1000000.0) > max_time_sec) {
      break;
    }

    if (!s) {
      LOG(INFO) << "Failed on run " << i;
      return s;
    }

    if (sleep_sec > 0) {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
    }
  }

  std::stringstream stream;
  stat.OutputToStream(&stream);
  LOG(INFO) << stream.str();

  return true;
}

DEFINE_string(device, "CPU", "Device [CPU|NEON|OPENCL]");
DEFINE_string(input_node, "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(output_node, "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(input_shape, "", "input shape, separated by colon and comma");
DEFINE_string(output_shape, "", "output shape, separated by colon and comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_int32(max_num_runs, 100, "number of runs max");
DEFINE_string(max_time, "10.0", "length to run max");
DEFINE_string(inference_delay, "-1", "delay between runs in seconds");
DEFINE_string(inter_benchmark_delay, "-1",
              "delay between benchmarks in seconds");
DEFINE_string(benchmark_name, "", "benchmark name");
DEFINE_bool(show_run_order, true, "whether to list stats by run order");
DEFINE_int32(run_order_limit, 0, "how many items to show by run order");
DEFINE_bool(show_time, true, "whether to list stats by time taken");
DEFINE_int32(time_limit, 10, "how many items to show by time taken");
DEFINE_bool(show_memory, false, "whether to list stats by memory used");
DEFINE_int32(memory_limit, 10, "how many items to show by memory used");
DEFINE_bool(show_type, true, "whether to list stats by op type");
DEFINE_bool(show_summary, true, "whether to show a summary of the stats");
DEFINE_bool(show_flops, true, "whether to estimate the model's FLOPs");
DEFINE_int32(warmup_runs, 1, "how many runs to initialize model");
DEFINE_string(model_data_file, "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy, 1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");

int Main(int argc, char **argv) {
  MACE_CHECK(FLAGS_device != "HEXAGON",
             "Model benchmark tool do not support DSP.");
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Benchmark name: [" << FLAGS_benchmark_name << "]";
  LOG(INFO) << "Device: [" << FLAGS_device << "]";
  LOG(INFO) << "gpu_perf_hint: [" << FLAGS_gpu_perf_hint << "]";
  LOG(INFO) << "gpu_priority_hint: [" << FLAGS_gpu_priority_hint << "]";
  LOG(INFO) << "omp_num_threads: [" << FLAGS_omp_num_threads << "]";
  LOG(INFO) << "cpu_affinity_policy: [" << FLAGS_cpu_affinity_policy << "]";
  LOG(INFO) << "Input node: [" << FLAGS_input_node<< "]";
  LOG(INFO) << "Input shapes: [" << FLAGS_input_shape << "]";
  LOG(INFO) << "Output node: [" << FLAGS_output_node<< "]";
  LOG(INFO) << "output shapes: [" << FLAGS_output_shape << "]";
  LOG(INFO) << "Warmup runs: [" << FLAGS_warmup_runs << "]";
  LOG(INFO) << "Num runs: [" << FLAGS_max_num_runs << "]";
  LOG(INFO) << "Inter-inference delay (seconds): ["
            << FLAGS_inference_delay << "]";
  LOG(INFO) << "Inter-benchmark delay (seconds): ["
            << FLAGS_inter_benchmark_delay << "]";

  const int64_t inter_inference_sleep_seconds =
      std::strtol(FLAGS_inference_delay.c_str(), nullptr, 10);
  const int64_t inter_benchmark_sleep_seconds =
      std::strtol(FLAGS_inter_benchmark_delay.c_str(), nullptr, 10);
  const double max_benchmark_time_seconds =
      std::strtod(FLAGS_max_time.c_str(), nullptr);

  std::unique_ptr<StatSummarizer> stats;

  StatSummarizerOptions stats_options;
  stats_options.show_run_order = FLAGS_show_run_order;
  stats_options.run_order_limit = FLAGS_run_order_limit;
  stats_options.show_time = FLAGS_show_time;
  stats_options.time_limit = FLAGS_time_limit;
  stats_options.show_memory = FLAGS_show_memory;
  stats_options.memory_limit = FLAGS_memory_limit;
  stats_options.show_type = FLAGS_show_type;
  stats_options.show_summary = FLAGS_show_summary;
  stats.reset(new StatSummarizer(stats_options));

  mace::DeviceType device_type = ParseDeviceType(FLAGS_device);

  // config runtime
  mace::SetOpenMPThreadPolicy(
      FLAGS_omp_num_threads,
      static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
  if (device_type == DeviceType::OPENCL) {
    mace::SetGPUHints(
        static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
        static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
  }

  std::vector<std::string> input_names =
      str_util::Split(FLAGS_input_node, ',');
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

  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
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
      LOG(INFO) << "Open input file failed";
      return -1;
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
    outputs[output_names[i]] = mace::MaceTensor(output_shape_vec[i],
                                                buffer_out);
  }

  // Init model
  LOG(INFO) << "Run init";
  std::unique_ptr<mace::MaceEngine> engine_ptr(
      new mace::MaceEngine(&net_def, device_type, input_names, output_names));
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }

  LOG(INFO) << "Warm up";

  int64_t warmup_time_us = 0;
  int64_t num_warmup_runs = 0;
  if (FLAGS_warmup_runs > 0) {
    bool status =
        Run(engine_ptr.get(), inputs, &outputs, nullptr,
            FLAGS_warmup_runs, -1.0,
            inter_inference_sleep_seconds, &warmup_time_us, &num_warmup_runs);
    if (!status) {
      LOG(ERROR) << "Failed at warm up run";
    }
  }

  if (inter_benchmark_sleep_seconds > 0) {
    std::this_thread::sleep_for(
        std::chrono::seconds(inter_benchmark_sleep_seconds));
  }
  int64_t no_stat_time_us = 0;
  int64_t no_stat_runs = 0;
  bool status =
      Run(engine_ptr.get(), inputs, &outputs,
          nullptr, FLAGS_max_num_runs, max_benchmark_time_seconds,
          inter_inference_sleep_seconds, &no_stat_time_us, &no_stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal no-stat run";
  }

  int64_t stat_time_us = 0;
  int64_t stat_runs = 0;
  status = Run(engine_ptr.get(), inputs, &outputs,
               stats.get(), FLAGS_max_num_runs, max_benchmark_time_seconds,
               inter_inference_sleep_seconds, &stat_time_us, &stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal stat run";
  }

  LOG(INFO) << "Average inference timings in us: "
            << "Warmup: "
            << (FLAGS_warmup_runs > 0 ? warmup_time_us / FLAGS_warmup_runs : 0)
            << ", " << "no stats: " << no_stat_time_us / no_stat_runs << ", "
            << "with stats: " << stat_time_us / stat_runs;

  stats->PrintOperatorStats();

  return 0;
}

}  // namespace benchmark
}  // namespace mace

int main(int argc, char **argv) { mace::benchmark::Main(argc, argv); }

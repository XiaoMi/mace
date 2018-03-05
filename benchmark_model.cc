//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"
#include "benchmark/stat_summarizer.h"

#include <cstdlib>
#include <fstream>
#include <thread>
#include <numeric>
#include <sys/time.h>

namespace mace {
namespace MACE_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}
}

namespace mace {
namespace str_util {

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    result.push_back(tmp.data());
    size_t next_offset = tmp.find(delims);
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

namespace benchmark {

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

bool RunInference(MaceEngine *engine,
                  const float *input,
                  const std::vector<int64_t> &input_shape,
                  float *output,
                  StatSummarizer *summarizer,
                  int64_t *inference_time_us) {
  RunMetadata run_metadata;
  RunMetadata *run_metadata_ptr = nullptr;
  if (summarizer) {
    run_metadata_ptr = &run_metadata;
  }
  const int64_t start_time = NowMicros();
  bool s = engine->Run(input, input_shape, output, run_metadata_ptr);
  const int64_t end_time = NowMicros();

  if (!s) {
    LOG(ERROR) << "Error during inference.";
    return s;
  }
  *inference_time_us = end_time - start_time;

  if (summarizer != nullptr) {
    summarizer->ProcessMetadata(run_metadata);
  }

  return true;
}

bool Run(MaceEngine *engine,
         const float *input,
         const std::vector<int64_t> &input_shape,
         float *output,
         StatSummarizer *summarizer,
         int num_runs,
         double max_time_sec,
         int64_t sleep_sec,
         int64_t *total_time_us,
         int64_t *actual_num_runs) {
  *total_time_us = 0;

  LOG(INFO) << "Running benchmark for max " << num_runs << " iterators, max ";
  LOG(INFO) << max_time_sec << " seconds ";
  LOG(INFO) << (summarizer != nullptr ? "with " : "without ");
  LOG(INFO) << "detailed stat logging, with " << sleep_sec;
  LOG(INFO) << "s sleep between inferences";

  Stat<int64_t> stat;

  bool util_max_time = (num_runs <= 0);
  for (int i = 0; util_max_time || i < num_runs; ++i) {
    int64_t inference_time_us = 0;
    bool s = RunInference(engine, input, input_shape, output, summarizer, &inference_time_us);
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

DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_string(device, "CPU", "Device [CPU|OPENCL]");
DEFINE_string(input_shape, "", "input shape, separated by comma");
DEFINE_string(output_shape, "", "output shape, separated by comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_int32(max_num_runs, 100, "number of runs max");
DEFINE_string(max_time, "10.0", "length to run max");
DEFINE_string(inference_delay, "-1", "delay between runs in seconds");
DEFINE_string(inter_benchmark_delay, "-1", "delay between benchmarks in seconds");
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
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");

int Main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> input_layer_shapes =
      str_util::Split(FLAGS_input_shape, ',');
  std::vector<int64_t> input_shape;
  mace::str_util::SplitAndParseToInts(FLAGS_input_shape, ',', &input_shape);
  std::vector<std::string> output_layer_shapes =
      str_util::Split(FLAGS_output_shape, ',');
  std::vector<int64_t> output_shape;
  mace::str_util::SplitAndParseToInts(FLAGS_input_shape, ',', &output_shape);

  LOG(INFO) << "Benchmark name: [" << FLAGS_benchmark_name << "]";
  LOG(INFO) << "Device: [" << FLAGS_device << "]";
  LOG(INFO) << "Input shapes: [" << FLAGS_input_shape << "]";
  LOG(INFO) << "output shapes: [" << FLAGS_output_shape << "]";
  LOG(INFO) << "Warmup runs: [" << FLAGS_warmup_runs << "]";
  LOG(INFO) << "Num runs: [" << FLAGS_max_num_runs << "]";
  LOG(INFO) << "Inter-inference delay (seconds): [" << FLAGS_inference_delay << "]";
  LOG(INFO) << "Inter-benchmark delay (seconds): [" << FLAGS_inter_benchmark_delay << "]";

  const long int inter_inference_sleep_seconds =
      std::strtol(FLAGS_inference_delay.c_str(), nullptr, 10);
  const long int inter_benchmark_sleep_seconds =
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

  DeviceType device_type = CPU;
  if(FLAGS_device == "OPENCL") {
    device_type = OPENCL;
  }

  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);

  int64_t input_size = std::accumulate(input_shape.begin(),
                                       input_shape.end(), 1, std::multiplies<int64_t>());
  int64_t output_size = std::accumulate(output_shape.begin(),
                                        output_shape.end(), 1, std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> output_data(new float[output_size]);

  // load input
  std::ifstream in_file(FLAGS_input_file, std::ios::in | std::ios::binary);
  if (in_file.is_open()) {
    in_file.read(reinterpret_cast<char *>(input_data.get()),
                 input_size * sizeof(float));
    in_file.close();
  } else {
    LOG(INFO) << "Open input file failed";
    return -1;
  }

  // Init model
  LOG(INFO) << "Run init";
  mace::MaceEngine engine(&net_def, device_type);
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }

  LOG(INFO) << "Warm up";

  int64_t warmup_time_us = 0;
  int64_t num_warmup_runs = 0;
  if (FLAGS_warmup_runs > 0) {
    bool status =
        Run(&engine, input_data.get(), input_shape, output_data.get(),
            nullptr, FLAGS_warmup_runs, -1.0,
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
      Run(&engine, input_data.get(), input_shape, output_data.get(),
          nullptr, FLAGS_max_num_runs, max_benchmark_time_seconds,
          inter_inference_sleep_seconds, &no_stat_time_us, &no_stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal no-stat run";
  }

  int64_t stat_time_us = 0;
  int64_t stat_runs = 0;
  status = Run(&engine, input_data.get(), input_shape, output_data.get(),
               stats.get(), FLAGS_max_num_runs, max_benchmark_time_seconds,
               inter_inference_sleep_seconds, &stat_time_us, &stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal stat run";
  }

  LOG(INFO) << "Average inference timings in us: ";
  LOG(INFO) << "Warmup: ";
  LOG(INFO) << (FLAGS_warmup_runs > 0 ? warmup_time_us / FLAGS_warmup_runs : 0) << ", ";
  LOG(INFO) << "no stats: " << no_stat_time_us / no_stat_runs << ", ";
  LOG(INFO) << "with stats: " << stat_time_us / stat_runs;

  stats->PrintOperatorStats();

  return 0;
}

}  //  namespace benchmark
}  //  namespace mace

int main(int argc, char **argv) { mace::benchmark::Main(argc, argv); }

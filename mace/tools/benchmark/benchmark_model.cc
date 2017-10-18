//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/net.h"
#include "mace/tools/benchmark/stat_summarizer.h"
#include "mace/utils/command_line_flags.h"
#include "mace/utils/utils.h"

#include <fstream>
#include <thread>

namespace mace {
namespace str_util {

std::vector<std::string> Split(const string &str, char delims) {
  std::vector<std::string> result;
  string tmp = str;
  while (!tmp.empty()) {
    result.push_back(tmp.data());
    size_t next_offset = tmp.find(delims);
    if (next_offset == string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}

bool SplitAndParseToInts(const string &str,
                         char delims,
                         std::vector<index_t> *result) {
  string tmp = str;
  while (!tmp.empty()) {
    index_t dim = atoi(tmp.data());
    result->push_back(dim);
    size_t next_offset = tmp.find(delims);
    if (next_offset == string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

}  //  namespace str_util

namespace benchmark {

bool RunInference(NetBase *net,
                  StatSummarizer *summarizer,
                  int64_t *inference_time_us) {
  RunMetadata run_metadata;
  RunMetadata *run_metadata_ptr = nullptr;
  if (summarizer) {
    run_metadata_ptr = &run_metadata;
  }
  const int64_t start_time = NowInMicroSec();
  bool s = net->Run(run_metadata_ptr);
  const int64_t end_time = NowInMicroSec();

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

bool Run(NetBase *net,
         StatSummarizer *summarizer,
         int num_runs,
         double max_time_sec,
         int64_t sleep_sec,
         int64_t *total_time_us,
         int64_t *actual_num_runs) {
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
    bool s = RunInference(net, summarizer, &inference_time_us);
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

int Main(int argc, char **argv) {
  std::string model_file = "/data/local/tmp/mobi_mace.pb";
  std::string device = "CPU";
  std::string input_layer_string = "input:0";
  std::string input_layer_shape_string = "1,224,224,3";
  std::string input_layer_type_string = "float";
  std::string input_layer_files_string = "";
  std::string output_layer_string = "output:0";
  int max_num_runs = 10;
  std::string max_time = "10.0";
  std::string inference_delay = "-1";
  std::string inter_benchmark_delay = "-1";
  int num_threads = -1;
  std::string benchmark_name = "";
  std::string output_prefix = "";
  bool show_sizes = false;
  bool show_run_order = true;
  int run_order_limit = 0;
  bool show_time = true;
  int time_limit = 10;
  bool show_memory = true;
  int memory_limit = 10;
  bool show_type = true;
  bool show_summary = true;
  bool show_flops = false;
  int warmup_runs = 2;

  std::vector<Flag> flag_list = {
      Flag("model_file", &model_file, "graph file name"),
      Flag("device", &device, "CPU/NEON"),
      Flag("input_layer", &input_layer_string, "input layer names"),
      Flag("input_layer_shape", &input_layer_shape_string, "input layer shape"),
      Flag("input_layer_type", &input_layer_type_string, "input layer type"),
      Flag("input_layer_files", &input_layer_files_string,
           "files to initialize the inputs with"),
      Flag("output_layer", &output_layer_string, "output layer name"),
      Flag("max_num_runs", &max_num_runs, "number of runs max"),
      Flag("max_time", &max_time, "length to run max"),
      Flag("inference_delay", &inference_delay,
           "delay between runs in seconds"),
      Flag("inter_benchmark_delay", &inter_benchmark_delay,
           "delay between benchmarks in seconds"),
      Flag("num_threads", &num_threads, "number of threads"),
      Flag("benchmark_name", &benchmark_name, "benchmark name"),
      Flag("output_prefix", &output_prefix, "benchmark output prefix"),
      Flag("show_sizes", &show_sizes, "whether to show sizes"),
      Flag("show_run_order", &show_run_order,
           "whether to list stats by run order"),
      Flag("run_order_limit", &run_order_limit,
           "how many items to show by run order"),
      Flag("show_time", &show_time, "whether to list stats by time taken"),
      Flag("time_limit", &time_limit, "how many items to show by time taken"),
      Flag("show_memory", &show_memory, "whether to list stats by memory used"),
      Flag("memory_limit", &memory_limit,
           "how many items to show by memory used"),
      Flag("show_type", &show_type, "whether to list stats by op type"),
      Flag("show_summary", &show_summary,
           "whether to show a summary of the stats"),
      Flag("show_flops", &show_flops, "whether to estimate the model's FLOPs"),
      Flag("warmup_runs", &warmup_runs, "how many runs to initialize model"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  std::vector<std::string> input_layers =
      str_util::Split(input_layer_string, ',');
  std::vector<std::string> input_layer_shapes =
      str_util::Split(input_layer_shape_string, ':');
  std::vector<string> input_layer_types =
      str_util::Split(input_layer_type_string, ',');
  std::vector<string> input_layer_files =
      str_util::Split(input_layer_files_string, ':');
  std::vector<string> output_layers = str_util::Split(output_layer_string, ',');
  if ((input_layers.size() != input_layer_shapes.size()) ||
      (input_layers.size() != input_layer_types.size())) {
    LOG(ERROR) << "There must be the same number of items in --input_layer,"
               << " --input_layer_shape, and --input_layer_type, for example"
               << " --input_layer=input1,input2 --input_layer_type=float,float "
               << " --input_layer_shape=1,224,224,4:1,20";
    LOG(ERROR) << "--input_layer=" << input_layer_string << " ("
               << input_layers.size() << " items)";
    LOG(ERROR) << "--input_layer_type=" << input_layer_type_string << " ("
               << input_layer_types.size() << " items)";
    LOG(ERROR) << "--input_layer_shape=" << input_layer_shape_string << " ("
               << input_layer_shapes.size() << " items)";
    return -1;
  }
  const size_t inputs_count = input_layers.size();

  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  LOG(INFO) << "Model file: [" << model_file << "]";
  LOG(INFO) << "Device: [" << device << "]";
  LOG(INFO) << "Input layers: [" << input_layer_string << "]";
  LOG(INFO) << "Input shapes: [" << input_layer_shape_string << "]";
  LOG(INFO) << "Input types: [" << input_layer_type_string << "]";
  LOG(INFO) << "Output layers: [" << output_layer_string << "]";
  LOG(INFO) << "Num runs: [" << max_num_runs << "]";
  LOG(INFO) << "Inter-inference delay (seconds): [" << inference_delay << "]";
  LOG(INFO) << "Inter-benchmark delay (seconds): [" << inter_benchmark_delay
            << "]";
  LOG(INFO) << "Num threads: [" << num_threads << "]";
  LOG(INFO) << "Benchmark name: [" << benchmark_name << "]";
  LOG(INFO) << "Output prefix: [" << output_prefix << "]";
  LOG(INFO) << "Show sizes: [" << show_sizes << "]";
  LOG(INFO) << "Warmup runs: [" << warmup_runs << "]";

  const long int inter_inference_sleep_seconds =
      std::strtol(inference_delay.c_str(), nullptr, 10);
  const long int inter_benchmark_sleep_seconds =
      std::strtol(inter_benchmark_delay.c_str(), nullptr, 10);
  const double max_benchmark_time_seconds =
      std::strtod(max_time.c_str(), nullptr);

  std::unique_ptr<StatSummarizer> stats;

  StatSummarizerOptions stats_options;
  stats_options.show_run_order = show_run_order;
  stats_options.run_order_limit = run_order_limit;
  stats_options.show_time = show_time;
  stats_options.time_limit = time_limit;
  stats_options.show_memory = show_memory;
  stats_options.memory_limit = memory_limit;
  stats_options.show_type = show_type;
  stats_options.show_summary = show_summary;
  stats.reset(new StatSummarizer(stats_options));

  // load model
  std::ifstream model_file_stream(model_file, std::ios::in | std::ios::binary);
  if (!model_file_stream.is_open()) {
    LOG(ERROR) << "model file open failed";
    return -1;
  }
  NetDef net_def;
  net_def.ParseFromIstream(&model_file_stream);
  model_file_stream.close();

  Workspace ws;
  ws.LoadModelTensor(net_def, DeviceType::CPU);
  // Load inputs
  for (size_t i = 0; i < inputs_count; ++i) {
    Tensor *input_tensor =
        ws.CreateTensor(input_layers[i], GetDeviceAllocator(DeviceType::CPU), DT_FLOAT);
    vector<index_t> shapes;
    str_util::SplitAndParseToInts(input_layer_shapes[i], ',', &shapes);
    input_tensor->Resize(shapes);
    float *input_data = input_tensor->mutable_data<float>();

    // load input
    if (i < input_layer_files.size()) {
      std::ifstream in_file(input_layer_files[i],
                            std::ios::in | std::ios::binary);
      in_file.read(reinterpret_cast<char *>(input_data),
                   input_tensor->size() * sizeof(float));
      in_file.close();
    }
  }

  // create net
  DeviceType device_type;
  DeviceType_Parse(device, &device_type);
  auto net = CreateNet(net_def, &ws, device_type);

  int64_t warmup_time_us = 0;
  int64_t num_warmup_runs = 0;
  if (warmup_runs > 0) {
    bool status =
        Run(net.get(), nullptr, warmup_runs, -1.0,
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
      Run(net.get(), nullptr, max_num_runs, max_benchmark_time_seconds,
          inter_inference_sleep_seconds, &no_stat_time_us, &no_stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal no-stat run";
  }

  int64_t stat_time_us = 0;
  int64_t stat_runs = 0;
  status = Run(net.get(), stats.get(), max_num_runs, max_benchmark_time_seconds,
               inter_inference_sleep_seconds, &stat_time_us, &stat_runs);
  if (!status) {
    LOG(ERROR) << "Failed at normal stat run";
  }

  LOG(INFO) << "Average inference timings in us: "
            << "Warmup: "
            << (warmup_runs > 0 ? warmup_time_us / warmup_runs : 0) << ", "
            << "no stats: " << no_stat_time_us / no_stat_runs << ", "
            << "with stats: " << stat_time_us / stat_runs;

  stats->PrintOperatorStats();

  return 0;
}

}  //  namespace benchmark
}  //  namespace mace

int main(int argc, char **argv) { mace::benchmark::Main(argc, argv); }

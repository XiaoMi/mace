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
#include <sys/time.h>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

using namespace std;
using namespace mace;

namespace mace {
namespace MACE_MODEL_TAG {

extern const unsigned char *LoadModelData(const char *model_data_file);

extern void UnloadModelData(const unsigned char *model_data);

extern NetDef CreateNet(const unsigned char *model_data);

extern const std::string ModelChecksum();

}  // namespace MACE_MODEL_TAG
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

DEFINE_string(input_shape, "1,224,224,3", "input shape, separated by comma");
DEFINE_string(output_shape, "1,224,224,2", "output shape, separated by comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_string(output_file, "", "output file name");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0");
DEFINE_string(device, "OPENCL", "CPU/NEON/OPENCL/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "mace version: " << MaceVersion();
  LOG(INFO) << "mace git version: " << MaceGitVersion();
  LOG(INFO) << "model checksum: " << mace::MACE_MODEL_TAG::ModelChecksum();
  LOG(INFO) << "input_shape: " << FLAGS_input_shape;
  LOG(INFO) << "output_shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "device: " << FLAGS_device;
  LOG(INFO) << "round: " << FLAGS_round;

  vector<int64_t> input_shape_vec;
  vector<int64_t> output_shape_vec;
  ParseShape(FLAGS_input_shape, &input_shape_vec);
  ParseShape(FLAGS_output_shape, &output_shape_vec);

  // load model
  int64_t t0 = NowMicros();
  const unsigned char *model_data =
      mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);
  int64_t t1 = NowMicros();
  LOG(INFO) << "CreateNetDef latency: " << t1 - t0 << " us";
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(FLAGS_device);
  LOG(INFO) << "Runing with device type: " << device_type;
  int64_t input_size =
      std::accumulate(input_shape_vec.begin(), input_shape_vec.end(), 1,
                      std::multiplies<int64_t>());
  int64_t output_size =
      std::accumulate(output_shape_vec.begin(), output_shape_vec.end(), 1,
                      std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> output_data(new float[output_size]);

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

  // Init model
  LOG(INFO) << "Run init";
  t0 = NowMicros();
  mace::MaceEngine engine(&net_def, device_type);
  if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
    mace::MACE_MODEL_TAG::UnloadModelData(model_data);
  }
  t1 = NowMicros();
  init_micros += t1 - t0;
  LOG(INFO) << "Net init latency: " << t1 - t0 << " us";
  LOG(INFO) << "Total init latency: " << init_micros << " us";

  LOG(INFO) << "Warm up run";
  t0 = NowMicros();
  engine.Run(input_data.get(), input_shape_vec, output_data.get());
  t1 = NowMicros();
  LOG(INFO) << "1st warm up run latency: " << t1 - t0 << " us";

  if (FLAGS_round > 0) {
    LOG(INFO) << "Run model";
    t0 = NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < FLAGS_round; ++i) {
      engine.Run(input_data.get(), input_shape_vec, output_data.get());
      if (FLAGS_malloc_check_cycle >= 1 && i % FLAGS_malloc_check_cycle == 0) {
        LOG(INFO) << "=== check malloc info change #" << i << " ===";
        prev = LogMallinfoChange(prev);
      }
    }
    t1 = NowMicros();
    LOG(INFO) << "Averate latency: " << (t1 - t0) / FLAGS_round << " us";
  }

  if (output_data != nullptr) {
    ofstream out_file(FLAGS_output_file, ios::binary);
    out_file.write((const char *)(output_data.get()),
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    LOG(INFO) << "Write output file done.";
  } else {
    LOG(INFO) << "Output data is null";
  }
}

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
 *          --device=OPENCL
 */
#include <cstdlib>
#include <fstream>
#include <malloc.h>
#include <numeric>
#include <iostream>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#include "gflags/gflags.h"
#include "mace/core/public/mace.h"

using namespace std;
using namespace mace;

namespace mace {
namespace MACE_MODEL_TAG {

extern NetDef CreateNet();

extern const std::string ModelChecksum();

}
}

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

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
    std::cout << "Non-mmapped space allocated (bytes): " << curr.arena
              << ", diff: " << ((int64_t)curr.arena - (int64_t)prev.arena)
              << std::endl;
  }
  if (prev.ordblks != curr.ordblks) {
    std::cout << "Number of free chunks: " << curr.ordblks
              << ", diff: " << ((int64_t)curr.ordblks - (int64_t)prev.ordblks)
              << std::endl;
  }
  if (prev.smblks != curr.smblks) {
    std::cout << "Number of free fastbin blocks: " << curr.smblks
              << ", diff: " << ((int64_t)curr.smblks - (int64_t)prev.smblks)
              << std::endl;
  }
  if (prev.hblks != curr.hblks) {
    std::cout << "Number of mmapped regions: " << curr.hblks
              << ", diff: " << ((int64_t)curr.hblks - (int64_t)prev.hblks)
              << std::endl;
  }
  if (prev.hblkhd != curr.hblkhd) {
    std::cout << "Space allocated in mmapped regions (bytes): " << curr.hblkhd
              << ", diff: " << ((int64_t)curr.hblkhd - (int64_t)prev.hblkhd)
              << std::endl;
  }
  if (prev.usmblks != curr.usmblks) {
    std::cout << "Maximum total allocated space (bytes): " << curr.usmblks
              << ", diff: " << ((int64_t)curr.usmblks - (int64_t)prev.usmblks)
              << std::endl;
  }
  if (prev.fsmblks != curr.fsmblks) {
    std::cout << "Space in freed fastbin blocks (bytes): " << curr.fsmblks
              << ", diff: " << ((int64_t)curr.fsmblks - (int64_t)prev.fsmblks)
              << std::endl;
  }
  if (prev.uordblks != curr.uordblks) {
    std::cout << "Total allocated space (bytes): " << curr.uordblks
              << ", diff: "
              << ((int64_t)curr.uordblks - (int64_t)prev.uordblks)
              << std::endl;
  }
  if (prev.fordblks != curr.fordblks) {
    std::cout << "Total free space (bytes): " << curr.fordblks << ", diff: "
              << ((int64_t)curr.fordblks - (int64_t)prev.fordblks)
              << std::endl;
  }
  if (prev.keepcost != curr.keepcost) {
    std::cout << "Top-most, releasable space (bytes): " << curr.keepcost
              << ", diff: "
              << ((int64_t)curr.keepcost - (int64_t)prev.keepcost)
              << std::endl;
  }
  return curr;
}

DEFINE_string(input_shape, "1,224,224,3", "input shape, separated by comma");
DEFINE_string(output_shape, "1,224,224,2", "output shape, separated by comma");
DEFINE_string(input_file, "", "input file name");
DEFINE_string(output_file, "", "output file name");
DEFINE_string(device, "OPENCL", "CPU/NEON/OPENCL/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "mace version: " << MaceVersion() << std::endl
            << "mace git version: " << MaceGitVersion() << std::endl
            << "model checksum: " << mace::MACE_MODEL_TAG::ModelChecksum() << std::endl
            << "input_shape: " << FLAGS_input_shape << std::endl
            << "output_shape: " << FLAGS_output_shape << std::endl
            << "input_file: " << FLAGS_input_file << std::endl
            << "output_file: " << FLAGS_output_file << std::endl
            << "device: " << FLAGS_device << std::endl
            << "round: " << FLAGS_round << std::endl;

  vector<int64_t> input_shape_vec;
  vector<int64_t> output_shape_vec;
  ParseShape(FLAGS_input_shape, &input_shape_vec);
  ParseShape(FLAGS_output_shape, &output_shape_vec);

  // load model
  int64_t t0 = NowMicros();
  NetDef net_def = mace::MACE_MODEL_TAG::CreateNet();
  int64_t t1 = NowMicros();
  std::cout << "CreateNetDef duration: " << t1 - t0 << " us" << std::endl;
  int64_t init_micros = t1 - t0;

  DeviceType device_type = ParseDeviceType(FLAGS_device);
  std::cout << "Device Type" << device_type << std::endl;
  int64_t input_size = std::accumulate(input_shape_vec.begin(),
      input_shape_vec.end(), 1, std::multiplies<int64_t>());
  int64_t output_size = std::accumulate(output_shape_vec.begin(),
      output_shape_vec.end(), 1, std::multiplies<int64_t>());
  std::unique_ptr<float[]> input_data(new float[input_size]);
  std::unique_ptr<float[]> output_data(new float[output_size]);

  // load input
  ifstream in_file(FLAGS_input_file, ios::in | ios::binary);
  if (in_file.is_open()) {
    in_file.read(reinterpret_cast<char *>(input_data.get()),
                 input_size * sizeof(float));
    in_file.close();
  } else {
    std::cout << "Open input file failed" << std::endl;
    return -1;
  }

  // Init model
  std::cout << "Run init" << std::endl;
  t0 = NowMicros();
  mace::MaceEngine engine(&net_def, device_type);
  t1 = NowMicros();
  init_micros += t1 - t0;
  std::cout << "Net init duration: " << t1 - t0 << " us" << std::endl;

  std::cout << "Total init duration: " << init_micros << " us" << std::endl;

  std::cout << "Warm up" << std::endl;
  t0 = NowMicros();
  engine.Run(input_data.get(), input_shape_vec, output_data.get());
  t1 = NowMicros();
  std::cout << "1st warm up run duration: " << t1 - t0 << " us" << std::endl;

  if (FLAGS_round > 0) {
    std::cout << "Run model" << std::endl;
    t0 = NowMicros();
    struct mallinfo prev = mallinfo();
    for (int i = 0; i < FLAGS_round; ++i) {
      engine.Run(input_data.get(), input_shape_vec, output_data.get());
      if (FLAGS_malloc_check_cycle >= 1 && i % FLAGS_malloc_check_cycle == 0) {
        std::cout << "=== check malloc info change #" << i << " ===" << std::endl;
        prev = LogMallinfoChange(prev);
      }
    }
    t1 = NowMicros();
    std::cout << "Avg duration: " << (t1 - t0) / FLAGS_round << " us" << std::endl;
  }

  if (output_data != nullptr) {
    ofstream out_file(FLAGS_output_file, ios::binary);
    out_file.write((const char *) (output_data.get()),
                   output_size * sizeof(float));
    out_file.flush();
    out_file.close();
    std::cout << "Write output file done." << std::endl;
  } else {
    std::cout << "output data is null" << std::endl;
  }
}

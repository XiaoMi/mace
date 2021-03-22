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

#ifndef MACE_UTILS_TUNER_H_
#define MACE_UTILS_TUNER_H_

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/types.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/proto/mace.pb.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/timer.h"

namespace mace {

constexpr const char *kOpenClWindowSize = "MACE_OPENCL_QUEUE_WINDOW_SIZE";

inline bool GetTuningFromEnv() {
  std::string tuning;
  GetEnv("MACE_TUNING", &tuning);
  return tuning.size() == 1 && tuning[0] == '1';
}

inline unsigned int GetOpenclQueueWindowSizeFromEnv() {
  std::string str_size;
  GetEnv(kOpenClWindowSize, &str_size);
  unsigned int window_size = 0;
  if (str_size.size() > 0) {
    window_size = atoi(str_size.c_str());
  }
  return window_size;
}

template <typename param_type>
class Tuner {
 public:
  explicit Tuner(const std::string tuned_param_file_path = "",
      const unsigned char *param_byte_stream = nullptr,
      const size_t param_byte_stream_size = 0) :
      tuned_param_file_path_(tuned_param_file_path) {
    MACE_CHECK(DataTypeToEnum<param_type>::value == DT_UINT32,
               "Only support save uint32_t tuning parameter");
    GetEnv("MACE_RUN_PARAMETER_PATH", &path_);
    is_tuning_ = GetTuningFromEnv();
    if (is_tuning_) {
      unsigned int wnd_size = GetOpenclQueueWindowSizeFromEnv();
      param_table_[kOpenClWindowSize] = {wnd_size};
    }

    if (param_byte_stream != nullptr && param_byte_stream_size != 0) {
      if (CheckArrayCRC32(param_byte_stream,
                          static_cast<uint64_t>(param_byte_stream_size))) {
        ParseData(param_byte_stream, param_byte_stream_size - CRC32SIZE);
      } else {
        LOG(WARNING) << "CRC value of provided array is invalid";
      }
    } else {
      ReadRunParamters();
    }
  }

  ~Tuner() { WriteRunParameters(); }

  Tuner(const Tuner &) = delete;
  Tuner &operator=(const Tuner &) = delete;

  template <typename RetType>
  RetType TuneOrRun(
      const std::string param_key,
      const std::vector<param_type> &default_param,
      const std::function<std::vector<std::vector<param_type>>()>
          &param_generator,
      const std::function<RetType(const std::vector<param_type> &,
                                  Timer *,
                                  std::vector<param_type> *)> &func,
      Timer *timer) {
    std::string obfuscated_param_key = MACE_OBFUSCATE_SYMBOL(param_key);
    if (IsTuning() && param_generator != nullptr) {
      // tune
      std::vector<param_type> opt_param = default_param;
      RetType res = Tune<RetType>(param_generator, func, timer, &opt_param);
      VLOG(3) << "Tuning " << param_key
              << " result: " << MakeString(opt_param);
      param_table_[obfuscated_param_key] = opt_param;
      return res;
    } else {
      // run
      if (param_table_.find(obfuscated_param_key) != param_table_.end()) {
        VLOG(3) << param_key << ": "
                << MakeString(param_table_[obfuscated_param_key]);
        return func(param_table_[obfuscated_param_key], nullptr, nullptr);
      } else {
        return func(default_param, nullptr, nullptr);
      }
    }
  }

  unsigned int GetOpenclQueueWindowSize() {
    unsigned int window_size = 0;
    if (!IsTuning()
        && param_table_.find(kOpenClWindowSize) != param_table_.end()) {
      window_size = param_table_[kOpenClWindowSize][0];
    }
    return window_size;
  }

  bool IsTuning() {
    return is_tuning_;
  }

 private:
  void WriteRunParameters() {
    if (!path_.empty()) {
      mace::PairContainer container;
      for (auto &elem : param_table_) {
        mace::KVPair *kvp = container.add_pairs();
        kvp->set_key(elem.first);
        const std::vector<uint32_t> &params = elem.second;
        for (const uint32_t &param : params) {
          kvp->add_uint32s_value(param);
        }
      }
      int data_size = container.ByteSize();
      std::unique_ptr<char[]> buffer = make_unique<char[]>(
          data_size + CRC32SIZE);
      char *buffer_ptr = buffer.get();
      if (!container.SerializeToArray(buffer_ptr, data_size)) {
        LOG(WARNING) << "Serialize protobuf to array failed.";
        return;
      }
      uint32_t crc_of_content = CalculateCRC32(
          reinterpret_cast<const unsigned char*>(buffer_ptr),
          static_cast<uint64_t>(data_size));
      memcpy(buffer_ptr+data_size, &crc_of_content, CRC32SIZE);
      std::ofstream ofs(path_.c_str(), std::ios::binary | std::ios::out);
      if (ofs.is_open()) {
        ofs.write(buffer_ptr, data_size + CRC32SIZE);
        ofs.close();
      } else {
        LOG(WARNING) << "Write run parameter file failed.";
      }
    }
  }

  void ParseData(const unsigned char *data, size_t data_size) {
    mace::PairContainer container;
    container.ParseFromArray(data, static_cast<int>(data_size));
    int num_pairs = container.pairs_size();
    for (int i = 0; i < num_pairs; ++i) {
      const mace::KVPair &kv = container.pairs(i);
      const auto &params_field = kv.uint32s_value();
      param_table_.emplace(kv.key(),
          std::vector<uint32_t>(params_field.begin(), params_field.end()));
    }
  }

  void ReadRunParamters() {
    if (!tuned_param_file_path_.empty()) {
      std::unique_ptr<mace::port::ReadOnlyMemoryRegion> param_data =
        make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
      auto fs = GetFileSystem();
      auto status = fs->NewReadOnlyMemoryRegionFromFile(
          tuned_param_file_path_.c_str(), &param_data);
      if (status != MaceStatus::MACE_SUCCESS)  {
        LOG(WARNING) << "Failed to read tuned param file: "
                     << tuned_param_file_path_;
        return;
      } else {
        if (CheckArrayCRC32(
            reinterpret_cast<const unsigned char *>(param_data->data()),
            static_cast<uint64_t>(param_data->length()))) {
          ParseData(static_cast<const unsigned char *>(param_data->data()),
                    param_data->length() - CRC32SIZE);
        } else {
          LOG(WARNING) << "CRC value of " << tuned_param_file_path_
                       << "is invalid";
        }
      }
    } else {
      VLOG(1) << "There is no tuned parameters.";
    }
  }

  template <typename RetType>
  RetType Run(const std::function<RetType(const std::vector<param_type> &,
                                          Timer *,
                                          std::vector<param_type> *)> &func,
              const std::vector<param_type> &params,
              Timer *timer,
              int num_runs,
              double *time_us,
              std::vector<param_type> *tuning_result) {
    RetType res = 0;
    int iter = 0;
    int64_t total_time_us = 0;
    for (iter = 0; iter < num_runs; ++iter) {
      res = func(params, timer, tuning_result);
      total_time_us += timer->AccumulatedMicros();
      if ((iter >= 1 && total_time_us > 100000) || total_time_us > 200000) {
        ++iter;
        break;
      }
    }

    *time_us = total_time_us * 1.0 / iter;
    return res;
  }

  template <typename RetType>
  RetType Tune(const std::function<std::vector<std::vector<param_type>>()>
               &param_generator,
               const std::function<RetType(const std::vector<param_type> &,
                                           Timer *,
                                           std::vector<param_type> *)> &func,
               Timer *timer,
               std::vector<param_type> *opt_params) {
    RetType res = 0;
    double opt_time = std::numeric_limits<double>::max();
    auto params = param_generator();
    std::vector<param_type> tuning_result;
    for (auto param : params) {
      double tmp_time = 0.0;
      // warm up
      Run<RetType>(func, param, timer, 1, &tmp_time, &tuning_result);

      // run
      RetType tmp_res =
          Run<RetType>(func, param, timer, 10, &tmp_time, &tuning_result);

      // Check the execution time
      if (tmp_time < opt_time) {
        opt_time = tmp_time;
        *opt_params = tuning_result;
        res = tmp_res;
      }
    }
    return res;
  }

 private:
  std::string tuned_param_file_path_;
  std::string path_;
  std::unordered_map<std::string, std::vector<param_type>> param_table_;
  unsigned int opencl_queue_window_size_;
  bool is_tuning_;
};

}  // namespace mace

#endif  // MACE_UTILS_TUNER_H_

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

#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/timer.h"

namespace mace {

inline bool IsTuning() {
  std::string tuning;
  GetEnv("MACE_TUNING", &tuning);
  return tuning.size() == 1 && tuning[0] == '1';
}

template <typename param_type>
class Tuner {
 public:
  explicit Tuner(const std::string tuned_param_file_path = "",
      const unsigned char *param_byte_stream = nullptr,
      const size_t param_byte_stream_size = 0):
      tuned_param_file_path_(tuned_param_file_path) {
    GetEnv("MACE_RUN_PARAMETER_PATH", &path_);
    if (param_byte_stream != nullptr && param_byte_stream_size != 0) {
      ParseData(param_byte_stream, param_byte_stream_size);
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
    std::string obfucated_param_key = MACE_OBFUSCATE_SYMBOL(param_key);
    if (IsTuning() && param_generator != nullptr) {
      // tune
      std::vector<param_type> opt_param = default_param;
      RetType res = Tune<RetType>(param_generator, func, timer, &opt_param);
      VLOG(3) << "Tuning " << param_key
              << " retult: " << MakeString(opt_param);
      param_table_[obfucated_param_key] = opt_param;
      return res;
    } else {
      // run
      if (param_table_.find(obfucated_param_key) != param_table_.end()) {
        VLOG(3) << param_key << ": "
                << MakeString(param_table_[obfucated_param_key]);
        return func(param_table_[obfucated_param_key], nullptr, nullptr);
      } else {
        return func(default_param, nullptr, nullptr);
      }
    }
  }

 private:
  void WriteRunParameters() {
    if (!path_.empty()) {
      VLOG(3) << "Write tuning result to " << path_;
      std::ofstream ofs(path_.c_str(), std::ios::binary | std::ios::out);
      if (ofs.is_open()) {
        int64_t num_pramas = param_table_.size();
        ofs.write(reinterpret_cast<char *>(&num_pramas), sizeof(num_pramas));
        for (auto &kp : param_table_) {
          int32_t key_size = kp.first.size();
          ofs.write(reinterpret_cast<char *>(&key_size), sizeof(key_size));
          ofs.write(kp.first.c_str(), key_size);

          auto &params = kp.second;
          int32_t params_size = params.size() * sizeof(param_type);
          ofs.write(reinterpret_cast<char *>(&params_size),
                    sizeof(params_size));

          VLOG(3) << "Write tuning param: " << kp.first.c_str() << ": "
                  << MakeString(params);
          for (auto &param : params) {
            ofs.write(reinterpret_cast<char *>(&param), sizeof(params_size));
          }
        }
        ofs.close();
      } else {
        LOG(WARNING) << "Write run parameter file failed.";
      }
    }
  }

  void ParseData(const unsigned char *data, size_t data_size) {
    const size_t int_size = sizeof(int32_t);
    const size_t param_type_size = sizeof(param_type);

    size_t parsed_offset = 0;
    int64_t num_params = 0;
    memcpy(&num_params, data, sizeof(num_params));
    data += sizeof(num_params);
    parsed_offset += sizeof(num_params);
    while (num_params--) {
      int32_t key_size = 0;
      memcpy(&key_size, data, int_size);
      data += int_size;
      std::string key(key_size, ' ');
      memcpy(&key[0], data, key_size);
      data += key_size;
      parsed_offset += int_size + key_size;

      int32_t params_size = 0;
      memcpy(&params_size, data, int_size);
      data += int_size;
      parsed_offset += int_size;
      int32_t params_count = params_size / param_type_size;
      std::vector<param_type> params(params_count);
      for (int i = 0; i < params_count; ++i) {
        memcpy(&params[i], data, param_type_size);
        data += param_type_size;
        parsed_offset += param_type_size;
      }
      MACE_CHECK(parsed_offset <= data_size,
                 "Parsing tuned data out of range: ",
                 parsed_offset, " > ", data_size);
      param_table_.emplace(key, params);
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
        LOG(ERROR) << "Failed to read tuned param file: "
                   << tuned_param_file_path_;
        return;
      } else {
        ParseData(static_cast<const unsigned char *>(param_data->data()),
                  param_data->length());
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
};

}  // namespace mace

#endif  // MACE_UTILS_TUNER_H_

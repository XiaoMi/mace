//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_TUNER_H_
#define MACE_UTILS_TUNER_H_
#include <stdlib.h>
#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <limits>

#include "mace/core/logging.h"

namespace mace {

template<typename param_type>
class Tuner {
 public:
  static Tuner* Get() {
    static Tuner tuner;
    return &tuner;
  }
  void TuneOrRun(const std::string &param_key,
              const std::vector<param_type> &default_param,
              std::function<std::vector<std::vector<param_type>>()> param_generator,
              const std::function<void(const std::vector<param_type> &)> &func) {

    if (param_generator == nullptr) {
      // run
      if (param_table_.find(param_key) != param_table_.end()) {
        func(param_table_[param_key]);
      } else {
        func(default_param);
      }
    } else {
      // tune
      std::vector<param_type> opt_param = default_param;
      Tune(param_generator, func, opt_param);
      param_table_[param_key] = opt_param;
    }
  }

 private:
  Tuner() {
    path_ = getenv("MACE_RUN_PARAMTER_PATH");
    ReadRunParamters();
  }

  ~Tuner() {
    WriteRunParameters();
  }

  Tuner(const Tuner&) = delete;
  Tuner& operator=(const Tuner&) = delete;

  inline void WriteRunParameters() {
    if (path_ != nullptr) {
      std::ofstream ofs(path_, std::ios::binary | std::ios::out);
      if (ofs.is_open()) {
        for (auto &kp : param_table_) {
          int32_t key_size = kp.first.size() + 1;
          ofs.write(static_cast<char*>(&key_size), sizeof(key_size));
          ofs.write(&kp.first.c_str(), key_size);

          auto &params = kp.second;
          int32_t params_size = params.size() * sizeof(param_type);
          ofs.write(static_cast<char*>(&params_size), sizeof(params_size));
          for (auto &param : params) {
            ofs.write(&param, sizeof(params_size));
          }
        }
        ofs.close();
      } else {
        LOG(WARNING) << "Write run parameter file failed.";
      }
    }
  }

  inline void ReadRunParamters() {
    if (path_ != nullptr) {
      std::ifstream ifs(path_, std::ios::binary | std::ios::in);
      if (ifs.is_open()) {
        int32_t key_size = 0;
        int32_t params_size = 0;
        int32_t params_count = 0;
        while (!ifs.eof()) {
          ifs.read(static_cast<char*>(&key_size), sizeof(key_size));
          std::string key(key_size, '');
          ifs.read(&key[0], key_size);

          ifs.read(static_cast<char*>(&params_size), sizeof(params_size));
          params_count = params_size / sizeof(param_type);
          std::vector<param_type> params(params_count);
          for (int i = 0; i < params_count; ++i) {
            ifs.read(&params[i], sizeof(param_type));
          }
          param_table_.emplace(key, params);
        }
        ifs.close();
      } else {
        LOG(WARNING) << "Write run parameter file failed.";
      }
    }
  }

  inline void Tune(std::function<std::vector<std::vector<param_type>>()> param_generator,
                   const std::function<void(const std::vector<param_type> &)> &func,
                   std::vector<param_type> &opt_params) {
    double opt_time = std::numeric_limits<double>::max();
    auto params = param_generator();
    for (const auto &param: params) {
      auto start = std::chrono::high_resolution_clock::now();
      func(param);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration_time = end - start;

      // Check the execution time
      if (duration_time.count() < opt_time) {
        opt_time = duration_time.count();
        opt_params = param;
      }
    }
  }

 private:
  const char* path_;
  std::unordered_map<std::string, std::vector<param_type>> param_table_;
};

} //  namespace mace
#endif //  MACE_UTILS_TUNER_H_

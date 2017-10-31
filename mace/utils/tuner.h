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
#include <limits>

#include "mace/core/logging.h"
#include "mace/utils/utils.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {


template<typename param_type>
class Tuner {
 public:
  static Tuner* Get() {
    static Tuner tuner;
    return &tuner;
  }

  template <typename RetType>
  RetType TuneOrRun(const std::string param_key,
              const std::vector<param_type> &default_param,
              const std::function<std::vector<std::vector<param_type>>()> &param_generator,
              const std::function<RetType(const std::vector<param_type> &)> &func) {

    if (IsTuning()) {
      // tune
      std::vector<param_type> opt_param = default_param;
      RetType res = Tune<RetType>(param_generator, func, opt_param);
      param_table_[param_key] = opt_param;
      return res;
    } else {
      // run
      if (param_table_.find(param_key) != param_table_.end()) {
        return func(param_table_[param_key]);
      } else {
        return func(default_param);
      }
    }
  }

 private:
  Tuner() {
    path_ = getenv("MACE_RUN_PARAMETER_PATH");
    ReadRunParamters();
  }

  ~Tuner() {
    WriteRunParameters();
  }

  Tuner(const Tuner&) = delete;
  Tuner& operator=(const Tuner&) = delete;

  inline bool IsTuning() {
    const char *tuning = getenv("MACE_TUNING");
    return tuning != nullptr && strlen(tuning) == 1 && tuning[0] == '1';
  }

  inline void WriteRunParameters() {
    VLOG(0) << path_;
    if (path_ != nullptr) {
      std::ofstream ofs(path_, std::ios::binary | std::ios::out);
      if (ofs.is_open()) {
        size_t num_pramas = param_table_.size();
        ofs.write(reinterpret_cast<char *>(&num_pramas), sizeof(num_pramas));
        for (auto &kp : param_table_) {
          int32_t key_size = kp.first.size();
          ofs.write(reinterpret_cast<char *>(&key_size), sizeof(key_size));
          ofs.write(kp.first.c_str(), key_size);
          VLOG(0) << kp.first.c_str();

          auto &params = kp.second;
          int32_t params_size = params.size() * sizeof(param_type);
          ofs.write(reinterpret_cast<char*>(&params_size), sizeof(params_size));
          for (auto &param : params) {
            ofs.write(reinterpret_cast<char *>(&param), sizeof(params_size));
            VLOG(0) << param;
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
        size_t num_pramas = 0;
        ifs.read(reinterpret_cast<char *>(&num_pramas), sizeof(num_pramas));
        while (num_pramas--) {
          ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
          std::string key(key_size, ' ');
          ifs.read(&key[0], key_size);

          ifs.read(reinterpret_cast<char *>(&params_size), sizeof(params_size));
          params_count = params_size / sizeof(param_type);
          std::vector<param_type> params(params_count);
          for (int i = 0; i < params_count; ++i) {
            ifs.read(reinterpret_cast<char*>(&params[i]), sizeof(param_type));
          }
          param_table_.emplace(key, params);
        }
        ifs.close();
      } else {
        LOG(WARNING) << "Read run parameter file failed.";
      }
    }
  }

  template <typename RetType>
  inline RetType Run(const std::function<RetType(const std::vector<param_type> &)> &func,
                     const std::vector<param_type> &params,
                     int num_runs,
                     double &time_us) {
    RetType res;
    int64_t total_time_us = 0;
    const int64_t start_time = NowInMicroSec();
    for (int i = 0; i < num_runs; ++i) {
      res = func(params);
    }
    OpenCLRuntime::Get()->command_queue().finish();
    const int64_t end_time = NowInMicroSec();
    total_time_us += end_time - start_time;

    time_us = total_time_us * 1.0 / num_runs;
    return res;
  }

  template <typename RetType>
  inline RetType Tune(std::function<std::vector<std::vector<param_type>>()> param_generator,
                   const std::function<RetType(const std::vector<param_type> &)> &func,
                   std::vector<param_type> &opt_params) {
    RetType res;
    double opt_time = std::numeric_limits<double>::max();
    auto params = param_generator();
    for (const auto &param: params) {
      double tmp_time = 0.0;
      // warm up
      Run<RetType>(func, param, 2, tmp_time);

      // run
      RetType tmp_res = Run<RetType>(func, param, 10, tmp_time);

      // Check the execution time
      if (tmp_time < opt_time) {
        opt_time = tmp_time;
        opt_params = param;
        res = tmp_res;
      }
    }
    return res;
  }

 private:
  const char* path_;
  std::unordered_map<std::string, std::vector<param_type>> param_table_;
};

} //  namespace mace
#endif //  MACE_UTILS_TUNER_H_

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_TUNER_H_
#define MACE_UTILS_TUNER_H_
#include <stdlib.h>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>

#include "mace/utils/logging.h"
#include "mace/utils/timer.h"
#include "mace/utils/utils.h"

namespace mace {

extern bool GetTuningParams(const char *path,
    std::unordered_map<std::string, std::vector<unsigned int>> *param_table);

template <typename param_type>
class Tuner {
 public:
  static Tuner *Get() {
    static Tuner tuner;
    return &tuner;
  }

  inline bool IsTuning() {
    const char *tuning = getenv("MACE_TUNING");
    return tuning != nullptr && strlen(tuning) == 1 && tuning[0] == '1';
  }

  template <typename RetType>
  RetType TuneOrRun(
      const std::string param_key,
      const std::vector<param_type> &default_param,
      const std::function<std::vector<std::vector<param_type>>()>
          &param_generator,
      const std::function<RetType(const std::vector<param_type> &)> &func,
      Timer *timer) {
    std::string obfucated_param_key = param_key;
    ObfuscateString(&obfucated_param_key);
    if (IsTuning() && param_generator != nullptr) {
      // tune
      std::vector<param_type> opt_param = default_param;
      RetType res = Tune<RetType>(param_generator, func, timer, &opt_param);
      VLOG(1) << "Tuning result. "
              << param_key << ": " << internal::MakeString(opt_param);
      param_table_[obfucated_param_key] = opt_param;
      return res;
    } else {
      // run
      if (param_table_.find(obfucated_param_key) != param_table_.end()) {
        VLOG(1) << param_key << ": "
                << internal::MakeString(param_table_[param_key]);
        return func(param_table_[param_key]);
      } else {
        LOG(WARNING) << "Fallback to default parameter: " << param_key;
        return func(default_param);
      }
    }
  }

 private:
  Tuner() {
    path_ = getenv("MACE_RUN_PARAMETER_PATH");
    ReadRunParamters();
  }

  ~Tuner() { WriteRunParameters(); }

  Tuner(const Tuner &) = delete;
  Tuner &operator=(const Tuner &) = delete;

  inline void WriteRunParameters() {
    VLOG(1) << path_;
    if (path_ != nullptr) {
      std::ofstream ofs(path_, std::ios::binary | std::ios::out);
      if (ofs.is_open()) {
        int64_t num_pramas = param_table_.size();
        ofs.write(reinterpret_cast<char *>(&num_pramas), sizeof(num_pramas));
        for (auto &kp : param_table_) {
          int32_t key_size = kp.first.size();
          ofs.write(reinterpret_cast<char *>(&key_size), sizeof(key_size));
          ofs.write(kp.first.c_str(), key_size);
          VLOG(1) << kp.first.c_str();

          auto &params = kp.second;
          int32_t params_size = params.size() * sizeof(param_type);
          ofs.write(reinterpret_cast<char *>(&params_size),
                    sizeof(params_size));
          for (auto &param : params) {
            ofs.write(reinterpret_cast<char *>(&param), sizeof(params_size));
            VLOG(1) << param;
          }
        }
        ofs.close();
      } else {
        LOG(WARNING) << "Write run parameter file failed.";
      }
    }
  }

  inline void ReadRunParamters() {
    bool success = GetTuningParams(path_, &param_table_);
    if (!success) {
      LOG(WARNING) << "Get run parameter failed.";
    }
  }

  template <typename RetType>
  inline RetType Run(
      const std::function<RetType(const std::vector<param_type> &)> &func,
      const std::vector<param_type> &params,
      Timer *timer,
      int num_runs,
      double *time_us) {
    RetType res;
    int64_t total_time_us = 0;
    for (int i = 0; i < num_runs; ++i) {
      timer->StartTiming();
      res = func(params);
      timer->StopTiming();
      total_time_us += timer->ElapsedMicros();
    }

    *time_us = total_time_us * 1.0 / num_runs;
    return res;
  }

  template <typename RetType>
  inline RetType Tune(
      const std::function<std::vector<std::vector<param_type>>()>
          &param_generator,
      const std::function<RetType(const std::vector<param_type> &)> &func,
      Timer *timer,
      std::vector<param_type> *opt_params) {
    RetType res;
    double opt_time = std::numeric_limits<double>::max();
    auto params = param_generator();
    for (const auto &param : params) {
      double tmp_time = 0.0;
      // warm up
      Run<RetType>(func, param, timer, 2, &tmp_time);

      // run
      RetType tmp_res = Run<RetType>(func, param, timer, 10, &tmp_time);

      // Check the execution time
      if (tmp_time < opt_time) {
        opt_time = tmp_time;
        *opt_params = param;
        res = tmp_res;
      }
    }
    return res;
  }

 private:
  const char *path_;
  std::unordered_map<std::string, std::vector<param_type>> param_table_;
};

}  //  namespace mace
#endif  //  MACE_UTILS_TUNER_H_

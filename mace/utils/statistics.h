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

#ifndef MACE_UTILS_STATISTICS_H_
#define MACE_UTILS_STATISTICS_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mace/public/mace.h"
#include "mace/utils/string_util.h"

namespace mace {

class RunMetadata;

namespace benchmark {

// stat the number of multiply-accumulate(MAC)
int64_t StatMACs(const std::string &op_type,
                 const std::vector<int64_t> &filter_shape,
                 const std::vector<int64_t> &output_shape);

template <typename IntType>
std::string IntToString(const IntType v) {
  std::stringstream stream;
  stream << v;
  std::string src_str = stream.str();
  size_t size = src_str.size();
  size_t dst_size = size + ((size-1) / 3);
  if (src_str[0] == '-') {
    dst_size = size + ((size-2) / 3);
  }
  std::string result(dst_size, ',');
  size_t dst_idx = dst_size - 1;
  for (size_t src_idx = 0; src_idx < size; ++src_idx) {
    if ((src_idx % 3) != 0 || src_idx == 0 || dst_idx == 0) {
      result[dst_idx] = src_str[size - 1 - src_idx];
    } else {
      dst_idx -= 1;
      result[dst_idx] = src_str[size - 1 - src_idx];
    }
    dst_idx -= 1;
  }
  return result;
}

template <typename FloatType>
std::string FloatToString(const FloatType v, const int32_t precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << v;
  return stream.str();
}
// microseconds
template <typename T>
class TimeInfo {
 public:
  TimeInfo():round_(0), first_(0), curr_(0),
             min_(std::numeric_limits<T>::max()), max_(0),
             sum_(0), square_sum(0)
  {}

  int64_t round() const {
    return round_;
  }

  T first() const {
    return first_;
  }

  T sum() const {
    return sum_;
  }

  double avg() const {
    return round_ == 0 ? std::numeric_limits<double>::quiet_NaN() :
           sum_ * 1.0f / round_;
  }

  double std_deviation() const {
    if (round_ == 0 || min_ == max_) {
      return 0;
    }
    const double avg_value = avg();
    return std::sqrt(square_sum / round_ - avg_value * avg_value);
  }

  void UpdateTime(const T time) {
    if (round_ == 0) {
      first_ = time;
    }

    curr_ = time;
    min_ = std::min<T>(min_, time);
    max_ = std::max<T>(max_, time);

    sum_ += time;
    square_sum += static_cast<double>(time) * time;
    round_ += 1;
  }

  std::string ToString(const std::string &title) const {
    std::vector<std::string> header = {
        "round", "first(ms)", "curr(ms)",
        "min(ms)", "max(ms)",
        "avg(ms)", "std"
    };
    std::vector<std::vector<std::string>> data(1);
    data[0].push_back(IntToString(round_));
    data[0].push_back(FloatToString(first_ / 1000.0, 3));
    data[0].push_back(FloatToString(curr_ / 1000.0, 3));
    data[0].push_back(FloatToString(min_ / 1000.0, 3));
    data[0].push_back(FloatToString(max_ / 1000.0, 3));
    data[0].push_back(FloatToString(avg() / 1000.0, 3));
    data[0].push_back(FloatToString(std_deviation(), 3));
    return mace::string_util::StringFormatter::Table(title, header, data);
  }

 private:
  int64_t round_;
  T first_;
  T curr_;
  T min_;
  T max_;
  T sum_;
  double square_sum;
};

enum Metric {
  NAME,
  RUN_ORDER,
  COMPUTATION_TIME,
};

class OpStat {
 public:
  void StatMetadata(const RunMetadata &meta_data);

  void PrintStat() const;

 private:
  std::string StatByMetric(const Metric metric,
      const int top_limit) const;
  std::string StatByOpType() const;
  std::string StatByMACs() const;
  std::string Summary() const;

 private:
  struct Record{
    std::string name;
    std::string type;
    std::vector<std::vector<int64_t>> output_shape;
    ConvPoolArgs args;
    int64_t macs;
    int64_t order;
    TimeInfo<int64_t> start;
    TimeInfo<int64_t> rel_end;
    int64_t called_times;
  };

  std::map<std::string, Record> records_;
  TimeInfo<int64_t> total_time_;
};

}  // namespace benchmark
}  // namespace mace
#endif  // MACE_UTILS_STATISTICS_H_

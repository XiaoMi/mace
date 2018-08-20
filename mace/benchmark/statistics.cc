// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <algorithm>
#include <set>

#include "mace/benchmark/statistics.h"
#include "mace/utils/logging.h"
#include "mace/utils/string_util.h"

namespace mace {
namespace benchmark {

namespace {
std::string MetricToString(const Metric metric) {
  switch (metric) {
    case NAME:
      return "Name";
    case RUN_ORDER:
      return "Run Order";
    case COMPUTATION_TIME:
      return "Computation Time";
    default:
      return "";
  }
}

std::string PaddingTypeToString(int padding_type) {
  std::stringstream stream;
  switch (padding_type) {
    case 0: stream << "VALID"; break;
    case 1: stream << "SAME"; break;
    case 2: stream << "FULL"; break;
    default: stream << padding_type; break;
  }

  return stream.str();
}

std::string ShapeToString(
    const std::vector<std::vector<int64_t>> &output_shape) {
  if (output_shape.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    size_t dims_size = output_shape[i].size();
    for (size_t j = 0; j < dims_size; ++j) {
      stream << output_shape[i][j];
      if (j != dims_size - 1) {
        stream << ",";
      }
    }
    if (i != output_shape.size() - 1) {
      stream << ":";
    }
  }
  stream << "]";

  return stream.str();
}

template <typename T>
std::string VectorToString(const std::vector<T> &vec) {
  if (vec.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    stream << vec[i];
    if (i != vec.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";

  return stream.str();
}

}  // namespace

void OpStat::StatMetadata(const RunMetadata &meta_data) {
  if (meta_data.op_stats.empty()) {
    LOG(FATAL) << "Op metadata should not be empty";
  }
  int64_t order_idx = 0;
  int64_t total_time = 0;

  const int64_t first_op_start_time = meta_data.op_stats[0].stats.start_micros;

  for (auto &op_stat : meta_data.op_stats) {
    auto result = records_.emplace(op_stat.operator_name, Record());
    Record *record = &(result.first->second);

    if (result.second) {
      record->name = op_stat.operator_name;
      record->type = op_stat.type;
      record->args = op_stat.args;
      record->output_shape = op_stat.output_shape;
      record->order = order_idx;
      order_idx += 1;
    }
    record->start.UpdateTime(op_stat.stats.start_micros - first_op_start_time);
    int64_t run_time = op_stat.stats.end_micros - op_stat.stats.start_micros;
    record->rel_end.UpdateTime(run_time);
    record->called_times += 1;
    total_time += run_time;
  }
  total_time_.UpdateTime(total_time);
}

std::string OpStat::StatByMetric(const Metric metric,
                                 const int top_limit) const {
  if (records_.empty()) {
    return "";
  }
  // sort
  std::vector<Record> records;
  for (auto &record : records_) {
    records.push_back(record.second);
  }
  std::sort(records.begin(), records.end(),
            [=](const Record &lhs, const Record &rhs) {
              if (metric == RUN_ORDER) {
                return lhs.order < rhs.order;
              } else if (metric == NAME) {
                return lhs.name.compare(rhs.name) < 0;
              } else {
                return lhs.rel_end.avg() > rhs.rel_end.avg();
              }
            });

  // generate string
  std::string title = "Sort by " + MetricToString(metric);
  const std::vector<std::string> header = {
      "Node Type", "Start", "First", "Avg(ms)", "%", "cdf%",
      "Stride", "Pad", "Filter Shape", "Output Shape", "Dilation", "name"
  };
  std::vector<std::vector<std::string>> data;
  int count = std::min(top_limit, static_cast<int>(records.size()));
  if (top_limit <= 0) count = static_cast<int>(records.size());

  int64_t accumulate_time = 0;
  for (int i = 0; i < count; ++i) {
    Record &record = records[i];
    accumulate_time += record.rel_end.sum();

    std::vector<std::string> tuple;
    tuple.push_back(record.type);
    tuple.push_back(FloatToString(record.start.avg() / 1000.0f, 3));
    tuple.push_back(FloatToString(record.rel_end.first() / 1000.0f, 3));
    tuple.push_back(FloatToString(record.rel_end.avg() / 1000.0f, 3));
    tuple.push_back(
        FloatToString(record.rel_end.sum() * 100.f / total_time_.sum(), 3));
    tuple.push_back(
        FloatToString(accumulate_time * 100.f / total_time_.sum(), 3));
    tuple.push_back(VectorToString<int>(record.args.strides));
    if (record.args.padding_type != -1) {
      tuple.push_back(PaddingTypeToString(record.args.padding_type));
    } else {
      tuple.push_back(VectorToString<int>(record.args.paddings));
    }
    tuple.push_back(VectorToString<int64_t>(record.args.kernels));
    tuple.push_back(ShapeToString(record.output_shape));
    tuple.push_back(VectorToString<int>(record.args.dilations));
    tuple.push_back(record.name);
    data.emplace_back(tuple);
  }
  return mace::string_util::StringFormatter::Table(title, header, data);
}

std::string OpStat::StatByNodeType() const {
  if (records_.empty()) {
    return "";
  }
  const int64_t round = total_time_.round();
  int64_t total_time = 0;
  std::map<std::string, int64_t> type_time_map;
  std::map<std::string, int64_t> type_count_map;
  std::map<std::string, int64_t> type_called_times_map;
  std::set<std::string> node_types_set;
  for (auto &record : records_) {
    std::string node_type = record.second.type;
    node_types_set.insert(node_type);

    type_time_map[node_type] += record.second.rel_end.sum() / round;
    total_time += record.second.rel_end.sum() / round;
    type_count_map[node_type] += 1;
    type_called_times_map[node_type] += record.second.called_times / round;
  }
  std::vector<std::string> node_types(node_types_set.begin(),
                                      node_types_set.end());
  std::sort(node_types.begin(), node_types.end(),
            [&](const std::string &lhs, const std::string &rhs) {
              return type_time_map[lhs] > type_time_map[rhs];
            });

  std::string title = "Stat by node type";
  const std::vector<std::string> header = {
      "Node Type", "Count", "Avg(ms)", "%", "cdf%", "Called times"
  };

  float cdf = 0.0f;
  std::vector<std::vector<std::string>> data;
  for (auto type : node_types) {
    const float avg_time = type_time_map[type] / 1000.0f;
    const float percentage = type_time_map[type] * 100.0f / total_time;
    cdf += percentage;

    std::vector<std::string> tuple;
    tuple.push_back(type);
    tuple.push_back(IntToString(type_count_map[type]));
    tuple.push_back(FloatToString(avg_time, 3));
    tuple.push_back(FloatToString(percentage, 3));
    tuple.push_back(FloatToString(cdf, 3));
    tuple.push_back(IntToString(type_called_times_map[type]));
    data.emplace_back(tuple);
  }
  return mace::string_util::StringFormatter::Table(title, header, data);
}

std::string OpStat::Summary() const {
  std::stringstream stream;
  if (!records_.empty()) {
    stream << total_time_.ToString("Summary of Ops' Stat") << std::endl;
  }

  stream << records_.size() << " ops total." << std::endl;

  return stream.str();
}

void OpStat::PrintStat() const {
  std::stringstream stream;
  if (!records_.empty()) {
  // op stat by run order
    stream << StatByMetric(Metric::RUN_ORDER, 0) << std::endl;
    // top-10 op stat by time
    stream << StatByMetric(Metric::COMPUTATION_TIME, 10) << std::endl;
    // op stat by node type
    stream << StatByNodeType() << std::endl;
  }
  // Print summary
  stream << Summary();

  for (std::string line; std::getline(stream, line);) {
    LOG(INFO) << line;
  }
}

}  // namespace benchmark
}  // namespace mace

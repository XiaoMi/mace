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

#include <algorithm>
#include <numeric>
#include <functional>
#include <set>

#include "mace/utils/statistics.h"
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


int64_t StatMACs(const std::string &op_type,
                 const std::vector<int64_t> &filter_shape,
                 const std::vector<int64_t> &output_shape) {
  int64_t macs = 0;
  if (op_type == "Conv2D" || op_type == "Deconv2D") {
    macs = output_shape[0] * output_shape[1] * output_shape[2]
        * output_shape[3]
        * filter_shape[2] * filter_shape[3] * filter_shape[1];
  } else if (op_type == "MatMul") {
    macs = std::accumulate(output_shape.begin(),
                           output_shape.end(),
                           1,
                           std::multiplies<int64_t>())
        * filter_shape.back();
  } else if (op_type == "DepthwiseConv2d") {
    macs = output_shape[0] * output_shape[1] * output_shape[2]
        * output_shape[3] * filter_shape[0] * filter_shape[2] * filter_shape[3];
  } else if (op_type == "DepthwiseDeconv2d") {
    macs = output_shape[0] * output_shape[1] * output_shape[2]
        * output_shape[3] * filter_shape[2] * filter_shape[3];
  } else if (op_type == "FullyConnected") {
    macs = output_shape[0] * std::accumulate(filter_shape.begin(),
                                             filter_shape.end(),
                                             1,
                                             std::multiplies<int64_t>());
  } else if (op_type == "BatchNorm") {
    macs = std::accumulate(output_shape.begin(),
                           output_shape.end(),
                           1,
                           std::multiplies<int64_t>());
  } else if (op_type == "ResizeBilinear" || op_type == "ResizeBicubic") {
    macs = 3 * std::accumulate(output_shape.begin(),
                               output_shape.end(),
                               1,
                               std::multiplies<int64_t>());
  } else if (op_type == "DynamicLSTM") {
    macs = output_shape[0] * (filter_shape[0] * filter_shape[1]
        + output_shape[1] * filter_shape[0] / 4);
  }
  return macs;
}

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
      record->macs =
          StatMACs(op_stat.type, op_stat.args.kernels, op_stat.output_shape[0]);
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
      "Op Type", "Start", "First", "Avg(ms)", "%", "cdf%", "GMACPS",
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
    tuple.push_back(FloatToString(
        record.macs < 1e-6 ? record.macs :
        (record.macs * 1e-3) / record.rel_end.avg(), 3));
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

std::string OpStat::StatByOpType() const {
  if (records_.empty()) {
    return "";
  }
  const int64_t round = total_time_.round();
  int64_t total_time = 0;
  std::map<std::string, int64_t> type_time_map;
  std::map<std::string, int64_t> type_macs_map;
  std::map<std::string, int64_t> type_count_map;
  std::map<std::string, int64_t> type_called_times_map;
  std::set<std::string> op_types_set;
  for (auto &record : records_) {
    std::string op_type = record.second.type;
    op_types_set.insert(op_type);

    type_time_map[op_type] += record.second.rel_end.sum() / round;
    type_macs_map[op_type] += record.second.macs;
    total_time += record.second.rel_end.sum() / round;
    type_count_map[op_type] += 1;
    type_called_times_map[op_type] += record.second.called_times / round;
  }
  std::vector<std::string> op_types(op_types_set.begin(),
                                    op_types_set.end());
  std::sort(op_types.begin(), op_types.end(),
            [&](const std::string &lhs, const std::string &rhs) {
              return type_time_map[lhs] > type_time_map[rhs];
            });

  std::string title = "Stat by Op Type";
  const std::vector<std::string> header = {
      "Op Type", "Count", "Avg(ms)", "%", "cdf%", "MACs",
      "GMACPS", "Called times"
  };

  float cdf = 0.0f;
  std::vector<std::vector<std::string>> data;
  for (auto type : op_types) {
    const float avg_time = type_time_map[type] / 1000.0f;
    const float percentage = type_time_map[type] * 100.0f / total_time;
    cdf += percentage;

    std::vector<std::string> tuple;
    tuple.push_back(type);
    tuple.push_back(IntToString(type_count_map[type]));
    tuple.push_back(FloatToString(avg_time, 3));
    tuple.push_back(FloatToString(percentage, 3));
    tuple.push_back(FloatToString(cdf, 3));
    tuple.push_back(IntToString(type_macs_map[type]));
    tuple.push_back(FloatToString(
        type_macs_map[type] < 1e-6 ? type_macs_map[type] :
        (type_macs_map[type] * 1e-3) / type_time_map[type], 3));
    tuple.push_back(IntToString(type_called_times_map[type]));
    data.emplace_back(tuple);
  }
  return mace::string_util::StringFormatter::Table(title, header, data);
}


std::string OpStat::StatByMACs() const {
  if (records_.empty()) {
    return "";
  }
  const int64_t round = total_time_.round();
  int64_t count = 0;
  for (auto &record : records_) {
    count += record.second.macs;
  }

  std::string title = "Stat by MACs(Multiply-Accumulation)";
  const std::vector<std::string> header = {
      "total", "round", "first(G/s)", "avg(G/s)", "std"
  };

  std::vector<std::vector<std::string>> data;
  std::vector<std::string> tuple;
  tuple.push_back(IntToString(count));
  tuple.push_back(IntToString(round));
  tuple.push_back(FloatToString((count * 1e-3) / total_time_.first(), 3));
  tuple.push_back(FloatToString((count * 1e-3) / total_time_.avg(), 3));
  tuple.push_back(FloatToString(total_time_.std_deviation(), 3));
  data.emplace_back(tuple);
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
    // op stat by op type
    stream << StatByOpType() << std::endl;
  }
  // print MACs statistics
  stream << StatByMACs();
  // Print summary
  stream << Summary();

  for (std::string line; std::getline(stream, line);) {
    LOG(INFO) << line;
  }
}

}  // namespace benchmark
}  // namespace mace

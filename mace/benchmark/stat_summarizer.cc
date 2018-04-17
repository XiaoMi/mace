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

#include "mace/benchmark/stat_summarizer.h"

#include <iomanip>
#include <iostream>
#include <queue>
#include <utility>

#include "mace/public/mace.h"
#include "mace/utils/logging.h"
#include "mace/core/types.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace benchmark {

StatSummarizer::StatSummarizer(const StatSummarizerOptions &options)
    : options_(options) {}

StatSummarizer::~StatSummarizer() {}

void StatSummarizer::Reset() {
  run_total_us_.Reset();
  memory_.Reset();
  details_.clear();
}

void StatSummarizer::ProcessMetadata(const RunMetadata &run_metadata) {
  int64_t curr_total_us = 0;
  int64_t mem_total = 0;

  if (run_metadata.op_stats.empty()) {
    std::cerr << "Runtime op stats should not be empty" << std::endl;
    abort();
  }
  int64_t first_node_start_us = run_metadata.op_stats[0].stats.start_micros;

  int node_num = 0;
  for (const auto &ops : run_metadata.op_stats) {
    std::string name = ops.operator_name;
    std::string op_type = ops.type;

    ++node_num;
    const int64_t curr_time = ops.stats.end_micros - ops.stats.start_micros;
    curr_total_us += curr_time;
    auto result = details_.emplace(name, Detail());
    Detail *detail = &(result.first->second);

    detail->start_us.UpdateStat(ops.stats.start_micros - first_node_start_us);
    detail->rel_end_us.UpdateStat(curr_time);

    // If this is the first pass, initialize some values.
    if (result.second) {
      detail->name = name;
      detail->type = op_type;
      detail->output_shape = ops.output_shape;
      detail->args = ops.args;

      detail->run_order = node_num;

      detail->times_called = 0;
    }

    ++detail->times_called;
  }

  run_total_us_.UpdateStat(curr_total_us);
  memory_.UpdateStat(mem_total);
}

std::string StatSummarizer::ShortSummary() const {
  std::stringstream stream;
  stream << "Timings (microseconds): ";
  run_total_us_.OutputToStream(&stream);
  stream << std::endl;

  stream << "Memory (bytes): ";
  memory_.OutputToStream(&stream);
  stream << std::endl;

  stream << details_.size() << " nodes observed" << std::endl;
  return stream.str();
}

std::ostream &InitField(std::ostream &stream, int width) {
  stream << std::right << std::setw(width) << std::fixed
         << std::setprecision(3);
  return stream;
}

std::string StatSummarizer::HeaderString(const std::string &title) const {
  std::stringstream stream;

  stream << "============================== " << title
         << " ==============================" << std::endl;

  InitField(stream, 24) << "[node type]";
  InitField(stream, 9) << "[start]";
  InitField(stream, 9) << "[first]";
  InitField(stream, 9) << "[avg ms]";
  InitField(stream, 9) << "[%]";
  InitField(stream, 9) << "[cdf%]";
  InitField(stream, 10) << "[mem KB]";
  InitField(stream, 10) << "[Name]";
  InitField(stream, 8) << "[stride]";
  InitField(stream, 10) << "[padding]";
  InitField(stream, 10) << "[dilation]";
  InitField(stream, 15) << "[kernel]";
  stream << std::right << std::setw(45) << "[output shape]";

  return stream.str();
}

std::string PaddingTypeToString(int padding_type) {
  std::stringstream stream;
  Padding type = static_cast<Padding>(padding_type);
  switch (type) {
    case VALID: stream << "VALID"; break;
    case SAME: stream << "SAME"; break;
    case FULL: stream << "FULL"; break;
    default: stream << padding_type; break;
  }

  return stream.str();
}

std::string ShapeToString(const std::vector<OutputShape> &output_shape) {
  if (output_shape.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < output_shape.size(); ++i) {
    const std::vector<index_t> &dims = output_shape[i].dims();
    for (int j = 0; j < dims.size(); ++j) {
      stream << dims[j];
      if (j != dims.size() - 1) {
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
  for (int i = 0; i < vec.size(); ++i) {
    stream << vec[i];
    if (i != vec.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";

  return stream.str();
}

std::string StatSummarizer::ColumnString(const StatSummarizer::Detail &detail,
                                         const int64_t cumulative_stat_on_node,
                                         const Stat<int64_t> &stat) const {
  const double start_ms = detail.start_us.avg() / 1000.0;
  const double first_time_ms = detail.rel_end_us.first() / 1000.0;
  const double avg_time_ms = detail.rel_end_us.avg() / 1000.0;
  const double percentage = detail.rel_end_us.sum() * 100.0 / stat.sum();
  const double cdf_percentage = (cumulative_stat_on_node * 100.0f) / stat.sum();

  std::stringstream stream;
  InitField(stream, 24) << detail.type;
  InitField(stream, 9) << start_ms;
  InitField(stream, 9) << first_time_ms;
  InitField(stream, 9) << avg_time_ms;
  InitField(stream, 8) << percentage << "%";
  InitField(stream, 8) << cdf_percentage << "%";
  InitField(stream, 10) << detail.mem_used.newest() / 1000.0;
  InitField(stream, 10) << detail.name;
  InitField(stream, 8) << VectorToString<int>(detail.args.strides);
  if (detail.args.padding_type != -1) {
    InitField(stream, 10) << PaddingTypeToString(detail.args.padding_type);
  } else {
    InitField(stream, 10) << VectorToString<int>(detail.args.paddings);
  }
  InitField(stream, 10) << VectorToString<int>(detail.args.dilations);
  InitField(stream, 15) << VectorToString<index_t>(detail.args.kernels);
  stream << std::right << std::setw(45) << ShapeToString(detail.output_shape);

  return stream.str();
}

void StatSummarizer::OrderNodesByMetric(
    SortingMetric metric, std::vector<const Detail *> *details) const {
  std::priority_queue<std::pair<std::string, const Detail *>> sorted_list;
  const int num_nodes = details_.size();

  for (const auto &det : details_) {
    const Detail *detail = &(det.second);

    std::stringstream stream;
    stream << std::setw(20) << std::right << std::setprecision(10)
           << std::fixed;

    switch (metric) {
      case BY_NAME:
        stream << detail->name;
        break;
      case BY_RUN_ORDER:
        stream << num_nodes - detail->run_order;
        break;
      case BY_TIME:
        stream << detail->rel_end_us.avg();
        break;
      case BY_MEMORY:
        stream << detail->mem_used.avg();
        break;
      case BY_TYPE:
        stream << detail->type;
        break;
      default:
        stream << "";
        break;
    }

    sorted_list.emplace(stream.str(), detail);
  }

  while (!sorted_list.empty()) {
    auto entry = sorted_list.top();
    sorted_list.pop();
    details->push_back(entry.second);
  }
}

void StatSummarizer::ComputeStatsByType(
    std::map<std::string, int64_t> *node_type_map_count,
    std::map<std::string, int64_t> *node_type_map_time,
    std::map<std::string, int64_t> *node_type_map_memory,
    std::map<std::string, int64_t> *node_type_map_times_called,
    int64_t *accumulated_us) const {
  int64_t run_count = run_total_us_.count();

  for (const auto &det : details_) {
    const std::string node_name = det.first;
    const Detail &detail = det.second;

    int64_t curr_time_val =
        static_cast<int64_t>(detail.rel_end_us.sum() / run_count);
    *accumulated_us += curr_time_val;

    int64_t curr_memory_val = detail.mem_used.newest();

    const std::string &node_type = detail.type;

    (*node_type_map_count)[node_type] += 1;
    (*node_type_map_time)[node_type] += curr_time_val;
    (*node_type_map_memory)[node_type] += curr_memory_val;
    (*node_type_map_times_called)[node_type] += detail.times_called / run_count;
  }
}

std::string StatSummarizer::GetStatsByNodeType() const {
  std::stringstream stream;

  stream << "============================== Summary by node type "
            "=============================="
         << std::endl;

  LOG(INFO) << "Number of nodes executed: " << details_.size() << std::endl;

  std::map<std::string, int64_t> node_type_map_count;
  std::map<std::string, int64_t> node_type_map_time;
  std::map<std::string, int64_t> node_type_map_memory;
  std::map<std::string, int64_t> node_type_map_times_called;
  int64_t accumulated_us = 0;

  ComputeStatsByType(&node_type_map_count, &node_type_map_time,
                     &node_type_map_memory, &node_type_map_times_called,
                     &accumulated_us);

  // Sort them.
  std::priority_queue<std::pair<int64_t, std::pair<std::string, int64_t>>>
      timings;
  for (const auto &node_type : node_type_map_time) {
    const int64_t mem_used = node_type_map_memory[node_type.first];
    timings.emplace(node_type.second,
                    std::pair<std::string, int64_t>(node_type.first, mem_used));
  }

  InitField(stream, 24) << "[Node type]";
  InitField(stream, 9) << "[count]";
  InitField(stream, 10) << "[avg ms]";
  InitField(stream, 11) << "[avg %]";
  InitField(stream, 11) << "[cdf %]";
  InitField(stream, 10) << "[mem KB]";
  InitField(stream, 10) << "[times called]";
  stream << std::endl;

  float cdf = 0.0f;
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();

    const std::string node_type = entry.second.first;
    const float memory = entry.second.second / 1000.0f;

    const int64_t node_type_total_us = entry.first;
    const float time_per_run_ms = node_type_total_us / 1000.0f;

    const float percentage =
        ((entry.first / static_cast<float>(accumulated_us)) * 100.0f);
    cdf += percentage;

    InitField(stream, 24) << node_type;
    InitField(stream, 9) << node_type_map_count[node_type];
    InitField(stream, 10) << time_per_run_ms;
    InitField(stream, 10) << percentage << "%";
    InitField(stream, 10) << cdf << "%";
    InitField(stream, 10) << memory;
    InitField(stream, 9) << node_type_map_times_called[node_type];
    stream << std::endl;
  }
  stream << std::endl;
  return stream.str();
}

std::string StatSummarizer::GetStatsByMetric(const std::string &title,
                                             SortingMetric sorting_metric,
                                             int num_stats) const {
  std::vector<const Detail *> details;
  OrderNodesByMetric(sorting_metric, &details);

  double cumulative_stat_on_node = 0;

  std::stringstream stream;
  stream << HeaderString(title) << std::endl;
  int stat_num = 0;
  for (auto detail : details) {
    ++stat_num;
    if (num_stats > 0 && stat_num > num_stats) {
      break;
    }

    cumulative_stat_on_node += detail->rel_end_us.sum();
    stream << ColumnString(*detail, cumulative_stat_on_node, run_total_us_)
           << std::endl;
  }
  stream << std::endl;
  return stream.str();
}

std::string StatSummarizer::GetOutputString() const {
  std::stringstream stream;
  if (options_.show_run_order) {
    stream << GetStatsByMetric("Run Order", BY_RUN_ORDER,
                               options_.run_order_limit);
  }
  if (options_.show_time) {
    stream << GetStatsByMetric("Top by Computation Time", BY_TIME,
                               options_.time_limit);
  }
  if (options_.show_memory) {
    stream << GetStatsByMetric("Top by Memory Use", BY_MEMORY,
                               options_.memory_limit);
  }
  if (options_.show_type) {
    stream << GetStatsByNodeType();
  }
  if (options_.show_summary) {
    stream << ShortSummary() << std::endl;
  }
  return stream.str();
}

void StatSummarizer::PrintOperatorStats() const {
  std::string output = GetOutputString();
  std::istringstream iss(output);
  for (std::string line; std::getline(iss, line);) {
    LOG(INFO) << line;
  }
}

}  // namespace benchmark
}  // namespace mace

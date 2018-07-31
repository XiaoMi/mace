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

#include "mace/utils/string_util.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>

namespace mace {
namespace string_util {

std::ostream &FormatRow(std::ostream &stream, int width) {
  stream << std::right << std::setw(width);
  return stream;
}

std::string StringFormatter::Table(
    const std::string &title,
    const std::vector<std::string> &header,
    const std::vector<std::vector<std::string>> &data) {
  if (header.empty()) return "";
  const size_t column_size = header.size();
  const size_t data_size = data.size();
  std::vector<int> max_column_len(header.size(), 0);
  for (size_t col_idx = 0; col_idx < column_size; ++col_idx) {
    max_column_len[col_idx] = std::max<int>(
        max_column_len[col_idx], static_cast<int>(header[col_idx].size()));
    for (size_t data_idx = 0; data_idx < data_size; ++data_idx) {
      if (col_idx < data[data_idx].size()) {
        max_column_len[col_idx] = std::max<int>(
            max_column_len[col_idx],
            static_cast<int>(data[data_idx][col_idx].size()));
      }
    }
  }
  const size_t row_length =
      std::accumulate(max_column_len.begin(), max_column_len.end(),
                      0, std::plus<size_t>())
          + 2 * column_size + column_size + 1;
  const std::string dash_line(row_length, '-');
  std::stringstream stream;
  stream << dash_line << std::endl;
  FormatRow(stream, static_cast<int>(row_length / 2 + title.size() / 2))
      << title << std::endl;
  stream << dash_line << std::endl;
  // format header
  stream << "|";
  for (size_t h_idx = 0; h_idx < column_size; ++h_idx) {
    stream << " ";
    FormatRow(stream, max_column_len[h_idx]) << header[h_idx];
    stream << " |";
  }
  stream << std::endl << dash_line << std::endl;
  // format data
  for (size_t data_idx = 0; data_idx < data_size; ++data_idx) {
    stream << "|";
    for (size_t h_idx = 0; h_idx < column_size; ++h_idx) {
      stream << " ";
      FormatRow(stream, max_column_len[h_idx]) << data[data_idx][h_idx];
      stream << " |";
    }
    stream << std::endl;
  }
  stream << dash_line << std::endl;
  return stream.str();
}

}  // namespace string_util
}  // namespace mace

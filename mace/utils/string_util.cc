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

std::string ObfuscateString(const std::string &src,
                            const std::string &lookup_table) {
  std::string dest;
  dest.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dest[i] = src[i] ^ lookup_table[i % lookup_table.size()];
  }
  return dest;
}

// ObfuscateString(ObfuscateString(str)) ==> str
std::string ObfuscateString(const std::string &src) {
  // Keep consistent with obfuscation in python tools
  return ObfuscateString(src, "Mobile-AI-Compute-Engine");
}

// Obfuscate synbol or path string
std::string ObfuscateSymbol(const std::string &src) {
  std::string dest = src;
  if (dest.empty()) {
    return dest;
  }
  dest[0] = src[0];  // avoid invalid symbol which starts from 0-9
  const std::string encode_dict =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
  for (size_t i = 1; i < src.size(); i++) {
    char ch = src[i];
    int idx;
    if (ch >= '0' && ch <= '9') {
      idx = ch - '0';
    } else if (ch >= 'a' && ch <= 'z') {
      idx = 10 + ch - 'a';
    } else if (ch >= 'A' && ch <= 'Z') {
      idx = 10 + 26 + ch - 'a';
    } else if (ch == '_') {
      idx = 10 + 26 + 26;
    } else {
      dest[i] = ch;
      continue;
    }
    // There is no collision if it's true for every char at every position
    dest[i] = encode_dict[(idx + i + 31) % encode_dict.size()];
  }
  return dest;
}

std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result.push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}
}  // namespace mace

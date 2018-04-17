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

#ifndef MACE_UTILS_COMMAND_LINE_FLAGS_H_
#define MACE_UTILS_COMMAND_LINE_FLAGS_H_

#include <string>
#include <vector>

namespace mace {

class Flag {
 public:
  Flag(const char *name, int *dst1, const std::string &usage_text);
  Flag(const char *name, int64_t *dst1, const std::string &usage_text);
  Flag(const char *name, bool *dst, const std::string &usage_text);
  Flag(const char *name, std::string *dst, const std::string &usage_text);
  Flag(const char *name, float *dst, const std::string &usage_text);

 private:
  friend class Flags;

  bool Parse(std::string arg, bool *value_parsing_ok) const;

  std::string name_;
  enum { TYPE_INT, TYPE_INT64, TYPE_BOOL, TYPE_STRING, TYPE_FLOAT } type_;
  int *int_value_;
  int64_t *int64_value_;
  bool *bool_value_;
  std::string *string_value_;
  float *float_value_;
  std::string usage_text_;
};

class Flags {
 public:
  // Parse the command line represented by argv[0, ..., (*argc)-1] to find flag
  // instances matching flags in flaglist[].  Update the variables associated
  // with matching flags, and remove the matching arguments from (*argc, argv).
  // Return true iff all recognized flag values were parsed correctly, and the
  // first remaining argument is not "--help".
  static bool Parse(int *argc, char **argv, const std::vector<Flag> &flag_list);

  // Return a usage message with command line cmdline, and the
  // usage_text strings in flag_list[].
  static std::string Usage(const std::string &cmdline,
                           const std::vector<Flag> &flag_list);
};

}  // namespace mace

#endif  // MACE_UTILS_COMMAND_LINE_FLAGS_H_

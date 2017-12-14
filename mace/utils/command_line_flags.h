//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_COMMAND_LINE_FLAGS_H
#define MACE_CORE_COMMAND_LINE_FLAGS_H

#include <string>
#include <vector>

namespace mace {

class Flag {
 public:
  Flag(const char *name, int *dst1, const std::string &usage_text);
  Flag(const char *name, long long *dst1, const std::string &usage_text);
  Flag(const char *name, bool *dst, const std::string &usage_text);
  Flag(const char *name, std::string *dst, const std::string &usage_text);
  Flag(const char *name, float *dst, const std::string &usage_text);

 private:
  friend class Flags;

  bool Parse(std::string arg, bool *value_parsing_ok) const;

  std::string name_;
  enum { TYPE_INT, TYPE_INT64, TYPE_BOOL, TYPE_STRING, TYPE_FLOAT } type_;
  int *int_value_;
  long long *int64_value_;
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

#endif  // MACE_CORE_COMMAND_LINE_FLAGS_H

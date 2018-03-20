//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/utils/command_line_flags.h"

#include <cstring>
#include <iomanip>

#include "mace/utils/logging.h"

namespace mace {
namespace utils {

bool StringConsume(const std::string &x, std::string *arg) {
  if ((arg->size() >= x.size()) &&
      (memcmp(arg->data(), x.data(), x.size()) == 0)) {
    *arg = arg->substr(x.size());
    return true;
  }
  return false;
}

bool ParseStringFlag(std::string arg,
                     std::string flag,
                     std::string *dst,
                     bool *value_parsing_ok) {
  *value_parsing_ok = true;
  if (StringConsume("--", &arg) && StringConsume(flag, &arg) &&
      StringConsume("=", &arg)) {
    *dst = arg;
    return true;
  }

  return false;
}

bool ParseInt32Flag(std::string arg,
                    std::string flag,
                    int32_t *dst,
                    bool *value_parsing_ok) {
  *value_parsing_ok = true;
  if (StringConsume("--", &arg) && StringConsume(flag, &arg) &&
      StringConsume("=", &arg)) {
    char extra;
    if (sscanf(arg.data(), "%d%c", dst, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

bool ParseInt64Flag(std::string arg,
                    std::string flag,
                    int64_t *dst,
                    bool *value_parsing_ok) {
  *value_parsing_ok = true;
  if (StringConsume("--", &arg) && StringConsume(flag, &arg) &&
      StringConsume("=", &arg)) {
    char extra;
    if (sscanf(arg.data(), "%lld%c", dst, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

bool ParseBoolFlag(std::string arg, std::string flag,
                   bool *dst, bool *value_parsing_ok) {
  *value_parsing_ok = true;
  if (StringConsume("--", &arg) && StringConsume(flag, &arg)) {
    if (arg.empty()) {
      *dst = true;
      return true;
    }

    if (arg == "=true") {
      *dst = true;
      return true;
    } else if (arg == "=false") {
      *dst = false;
      return true;
    } else {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
      return true;
    }
  }

  return false;
}

bool ParseFloatFlag(std::string arg,
                    std::string flag,
                    float *dst,
                    bool *value_parsing_ok) {
  *value_parsing_ok = true;
  if (StringConsume("--", &arg) && StringConsume(flag, &arg) &&
      StringConsume("=", &arg)) {
    char extra;
    if (sscanf(arg.data(), "%f%c", dst, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

}  // namespace utils

Flag::Flag(const char *name, int *dst, const std::string &usage_text)
    : name_(name), type_(TYPE_INT), int_value_(dst), usage_text_(usage_text) {}

Flag::Flag(const char *name, int64_t *dst, const std::string &usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_value_(dst),
      usage_text_(usage_text) {}

Flag::Flag(const char *name, bool *dst, const std::string &usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_value_(dst),
      usage_text_(usage_text) {}

Flag::Flag(const char *name, std::string *dst, const std::string &usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_value_(dst),
      usage_text_(usage_text) {}

Flag::Flag(const char *name, float *dst, const std::string &usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      float_value_(dst),
      usage_text_(usage_text) {}

bool Flag::Parse(std::string arg, bool *value_parsing_ok) const {
  bool result = false;
  if (type_ == TYPE_INT) {
    result = utils::ParseInt32Flag(arg, name_, int_value_, value_parsing_ok);
  } else if (type_ == TYPE_INT64) {
    result = utils::ParseInt64Flag(arg, name_, int64_value_, value_parsing_ok);
  } else if (type_ == TYPE_BOOL) {
    result = utils::ParseBoolFlag(arg, name_, bool_value_, value_parsing_ok);
  } else if (type_ == TYPE_STRING) {
    result = utils::ParseStringFlag(arg, name_,
                                    string_value_, value_parsing_ok);
  } else if (type_ == TYPE_FLOAT) {
    result = utils::ParseFloatFlag(arg, name_, float_value_, value_parsing_ok);
  }
  return result;
}

/*static*/ bool Flags::Parse(int *argc,
                             char **argv,
                             const std::vector<Flag> &flag_list) {
  bool result = true;
  std::vector<char *> unknown_flags;
  for (int i = 1; i < *argc; ++i) {
    if (std::string(argv[i]) == "--") {
      while (i < *argc) {
        unknown_flags.push_back(argv[i]);
        ++i;
      }
      break;
    }

    bool was_found = false;
    for (const Flag &flag : flag_list) {
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], &value_parsing_ok);
      if (!value_parsing_ok) {
        result = false;
      }
      if (was_found) {
        break;
      }
    }
    if (!was_found) {
      unknown_flags.push_back(argv[i]);
    }
  }
  // Passthrough any extra flags.
  int dst = 1;  // Skip argv[0]
  for (char *f : unknown_flags) {
    argv[dst++] = f;
  }
  argv[dst++] = nullptr;
  *argc = unknown_flags.size() + 1;
  return result && (*argc < 2 || strcmp(argv[1], "--help") != 0);
}

std::string Flags::Usage(const std::string &cmdline,
                         const std::vector<Flag> &flag_list) {
  std::stringstream usage_text;
  usage_text << "usage: " << cmdline << std::endl;

  if (!flag_list.empty()) {
    usage_text << "Flags: " << std::endl;
  }
  for (const Flag &flag : flag_list) {
    usage_text << "\t" << std::left << std::setw(30) << flag.name_;
    usage_text << flag.usage_text_ << std::endl;
  }
  return usage_text.str();
}

}  // namespace mace

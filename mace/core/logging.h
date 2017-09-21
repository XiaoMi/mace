//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_LOGGING_H_
#define MACE_CORE_LOGGING_H_

#include <limits>
#include <sstream>
#include <string>
#include <vector>

#undef ERROR

namespace mace {
const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

using std::string;

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream& ss,
                               const T& t,
                               const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return ss.str();
}

template <typename T>
string MakeString(const std::vector<T> &args) {
  std::stringstream ss;
  for (const T& arg: args) {
    ss << arg << ", ";
  }
  return ss.str();
}

// Specializations for already-a-string types.
template <>
inline string MakeString(const string& str) {
  return str;
}
inline string MakeString(const char* c_str) { return string(c_str); }

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MinVLogLevel();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _MACE_LOG_INFO \
  ::mace::internal::LogMessage(__FILE__, __LINE__, mace::INFO)
#define _MACE_LOG_WARNING \
  ::mace::internal::LogMessage(__FILE__, __LINE__, mace::WARNING)
#define _MACE_LOG_ERROR \
  ::mace::internal::LogMessage(__FILE__, __LINE__, mace::ERROR)
#define _MACE_LOG_FATAL ::mace::internal::LogMessageFatal(__FILE__, __LINE__)

#define _MACE_LOG_QFATAL _MACE_LOG_FATAL

#define LOG(severity) _MACE_LOG_##severity

#ifdef IS_MOBILE_PLAMACEORM
// Turn VLOG off when under mobile devices for considerations of binary size.
#define VLOG_IS_ON(lvl) ((lvl) <= 0)
#else
// Otherwise, Set MACE_CPP_MIN_VLOG_LEVEL environment to update minimum log
// level
// of VLOG
#define VLOG_IS_ON(lvl) ((lvl) <= ::mace::internal::LogMessage::MinVLogLevel())
#endif

#define VLOG(lvl)      \
  if (VLOG_IS_ON(lvl)) \
  ::mace::internal::LogMessage(__FILE__, __LINE__, mace::INFO)

// MACE_CHECK/MACE_ASSERT dies with a fatal error if condition is not true.
// MACE_ASSERT is controlled by NDEBUG ('-c opt' for bazel) while MACE_CHECK
// will be executed regardless of compilation mode.
// Therefore, it is safe to do things like:
//    MACE_CHECK(fp->Write(x) == 4)
//    MACE_CHECK(fp->Write(x) == 4, "Write failed")
// which are not correct for MACE_ASSERT.
#define MACE_CHECK(condition, ...)              \
  if (!(condition))                             \
  LOG(FATAL) << "Check failed: " #condition " " \
             << ::mace::internal::MakeString(__VA_ARGS__)

#ifndef NDEBUG
#define MACE_ASSERT(condition, ...)              \
  if (!(condition))                              \
  LOG(FATAL) << "Assert failed: " #condition " " \
             << ::mace::internal::MakeString(__VA_ARGS__)
#else
#define MACE_ASSERT(condition, ...) ((void)0)
#endif

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

#define MACE_CHECK_NOTNULL(val)                      \
  ::mace::internal::CheckNotNull(__FILE__, __LINE__, \
                                 "'" #val "' Must be non NULL", (val))

}  // namespace internal
}  // namespace mace

#endif  // MACE_CORE_LOGGING_H_

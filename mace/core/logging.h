#ifndef MACE_COMMON_LOGGING_H_
#define MACE_COMMON_LOGGING_H_

#ifdef __ANDROID__
#include <android/log.h>
#else
#include <cstdio>
#endif

namespace mace {

const int FATAL = 0;
const int ERROR = 1;
const int WARN = 2;
const int INFO = 3;
const int DEBUG = 4;
const int VERBOSE = 5;

namespace internal {

const char *kTag = "MACE";


#ifdef __ANDROID__

#define _MACE_LOG_FATAL \
  do { \
    __android_log_print(ANDROID_LOG_FATAL, mace::internal::kTag, __VA_ARGS__); \
    abort(); \
  } while (0)

#define _MACE_LOG_ERROR(...) \
  __android_log_print(ANDROID_LOG_ERROR, mace::internal::kTag, __VA_ARGS__)
#define _MACE_LOG_WARN(...) \
  __android_log_print(ANDROID_LOG_WARN, mace::internal::kTag, __VA_ARGS__)
#define _MACE_LOG_INFO(...) \
  __android_log_print(ANDROID_LOG_INFO, mace::internal::kTag, __VA_ARGS__)
#define _MACE_LOG_DEBUG(...) \
  __android_log_print(ANDROID_LOG_DEBUG, mace::internal::kTag, __VA_ARGS__)
#define _MACE_LOG_VERBOSE(...) \
  __android_log_print(ANDROID_LOG_VERBOSE, mace::internal::kTag, __VA_ARGS__)
  

#define LOG(severity, ...) _MACE_LOG_##severity(__VA_ARGS__)

#else // Non Android, just for tests

// TODO(heliangliang): Fix newline
#define LOG(severity, ...) \
  do { \
    printf(__VA_ARGS__); \
    printf("\n"); \
  } while (0)

#endif // __ANDROID__

} // namespace internal
} // namespace mace

#endif // MACE_COMMON_LOGGING_H_

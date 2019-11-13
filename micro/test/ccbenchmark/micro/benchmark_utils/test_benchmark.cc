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

#include "micro/benchmark_utils/test_benchmark.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/common/global_buffer.h"
#include "micro/port/api.h"

namespace micro {
namespace base {
template<typename T>
char *ToString(T value, char *buffer, char *end);
template<>
char *ToString(float value, char *buffer, char *end);
template<>
char *ToString(int32_t value, char *buffer, char *end);
template<>
char *ToString(int64_t value, char *buffer, char *end);
}  // namespace base

namespace testing {
namespace {
const int32_t kMaxBenchmarkNum = 200;

const int32_t kNameWidth = 50 + 1;
const int32_t kInt64ValueBufferLength = 21;
const int32_t kInt32ValueBufferLength = 12;
const int32_t kFloatValueBufferLength = 21;
void GetFixWidthStr(const char *input, char *output, const int32_t fix_width) {
  int32_t length = micro::base::strlen(input);
  if (length >= fix_width) {
    micro::base::memcpy(output, input, fix_width * sizeof(char));
  } else {
    micro::base::memcpy(output, input, length * sizeof(char));
    while (length < fix_width) {
      output[length++] = ' ';
    }
  }
  output[fix_width] = '\0';
}

void GetFixWidthStr(int32_t input, char *output, const int32_t fix_width) {
  char int_str[kInt32ValueBufferLength] = {0};
  micro::base::ToString(input, int_str, int_str + kInt32ValueBufferLength);
  GetFixWidthStr(int_str, output, fix_width);
}

void GetFixWidthStr(int64_t input, char *output, const int32_t fix_width) {
  char int_str[kInt64ValueBufferLength] = {0};
  micro::base::ToString(input, int_str, int_str + kInt64ValueBufferLength);
  GetFixWidthStr(int_str, output, fix_width);
}

void GetFixWidthStr(float input, char *output, const int32_t fix_width) {
  char int_str[kFloatValueBufferLength] = {0};
  micro::base::ToString(input, int_str, int_str + kFloatValueBufferLength);
  GetFixWidthStr(int_str, output, fix_width);
}

Benchmark *all_benchmarks[kMaxBenchmarkNum] = {NULL};
int32_t benchmark_size = 0;
int64_t bytes_processed;
int64_t macs_processed = 0;
int64_t accum_time = 0;
int64_t start_time = 0;

}  // namespace

Benchmark::Benchmark(const char *name, BenchmarkFunc *benchmark_func)
    : name_(name), benchmark_func_(benchmark_func) {
  Register();
}

void Benchmark::Run() {
  LOG(INFO) << "Benchmark::Run start, benchmark_size=" << benchmark_size;
  if (benchmark_size == 0) {
    return;
  }

  char benchmark_name[kNameWidth] = {0};
  GetFixWidthStr("Benchmark", benchmark_name, kNameWidth - 1);
  char time_name[kInt64ValueBufferLength] = {0};
  GetFixWidthStr("Time(ns)", time_name, kInt64ValueBufferLength - 1);
  char iterations_name[kInt32ValueBufferLength] = {0};
  GetFixWidthStr("Iterations", iterations_name, kInt32ValueBufferLength - 1);
  char input_mb_name[kFloatValueBufferLength] = {0};
  GetFixWidthStr("Input(MB/s)", input_mb_name, kFloatValueBufferLength - 1);
  LOG(CLEAN) << benchmark_name << "\t" << time_name << "\t" << iterations_name
             << "\t" << input_mb_name << "\t" << "GMACPS";
  LOG(CLEAN) << "--------------------------------------------------------------"
                "-------------------------------------------------------------";

  for (int32_t i = 0; i < benchmark_size; ++i) {
    Benchmark *b = all_benchmarks[i];
    int32_t iters;
    double seconds;
    b->Run(&iters, &seconds);
    float mbps = (bytes_processed * 1e-6) / seconds;
    // MACCs or other computations
    float gmacs = (macs_processed * 1e-9) / seconds;
    int64_t ns = static_cast<int64_t>(seconds * 1e9);

    char name_str[kNameWidth] = {0};
    GetFixWidthStr(b->name_, name_str, kNameWidth - 1);
    char ns_str[kInt64ValueBufferLength] = {0};
    GetFixWidthStr(ns / iters, ns_str, kInt64ValueBufferLength - 1);
    char iters_str[kInt32ValueBufferLength] = {0};
    GetFixWidthStr(iters, iters_str, kInt32ValueBufferLength - 1);
    char mbps_str[kFloatValueBufferLength] = {0};
    GetFixWidthStr(mbps, mbps_str, kFloatValueBufferLength - 1);
    char gmacs_str[kInt32ValueBufferLength] = {0};
    if (gmacs != 0) {
      GetFixWidthStr(gmacs, gmacs_str, kInt32ValueBufferLength - 1);
    } else {
      gmacs_str[0] = '-';
    }
    LOG(CLEAN) << name_str << "\t" << ns_str << "\t"
               << iters_str << "\t" << mbps_str << "\t" << gmacs_str;
  }
}

void Benchmark::Register() {
  MACE_ASSERT2(benchmark_size < kMaxBenchmarkNum,
               "benchmark_size is:", benchmark_size);
  all_benchmarks[benchmark_size++] = this;
}

void Benchmark::Run(int32_t *run_count, double *run_seconds) {
  static const int32_t kMinIters = 10;
  static const int32_t kMaxIters = 10000;
  static const double kMinTime = 0.5;
  int32_t iters = kMinIters;
  while (true) {
    bytes_processed = -1;
    macs_processed = 0;
    common::test::GetGlobalBuffer()->reset();
    RestartTiming();
    (*benchmark_func_)(iters);
    StopTiming();
    const double seconds = accum_time * 1e-6;
    if (seconds >= kMinTime || iters >= kMaxIters) {
      *run_count = iters;
      *run_seconds = seconds;
      return;
    }

    // Update number of iterations.
    // Overshoot by 100% in an attempt to succeed the next time.
    double multiplier = 2.0 * kMinTime / base::max(seconds, 1e-9);
    iters = base::min<int64_t>(multiplier * iters, kMaxIters);  // NOLINT
  }
}

void BytesProcessed(int64_t n) { bytes_processed = n; }
void MacsProcessed(int64_t n) { macs_processed = n; }
void RestartTiming() {
  accum_time = 0;
  start_time = port::api::NowMicros();
}
void StartTiming() {
  start_time = port::api::NowMicros();
}
void StopTiming() {
  if (start_time != 0) {
    accum_time += (port::api::NowMicros() - start_time);
    start_time = 0;
  }
}

}  // namespace testing
}  // namespace micro

extern "C" {
void BenchmarkRun() {
  micro::testing::Benchmark::Run();
}
}

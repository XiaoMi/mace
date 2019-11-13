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

#include "micro/rpc/stub/benchmark.h"
#include "micro/test/ccbenchmark/codegen/benchmark.h"

namespace micro {
namespace testing {

namespace {
const char kBenchmarkUri[] = benchmark_URI"&_dom=sdsp";
}  // namespace

Benchmark::Benchmark() :
    rpc::stub::BaseHandle(benchmark_open, benchmark_close, kBenchmarkUri) {}

void Benchmark::Run() {
  benchmark_run(remote_handle_);
}

}  // namespace testing
}  // namespace micro

void BenchmarkRun() {
  micro::testing::Benchmark benchmark;
  benchmark.Open();
  benchmark.Run();
  benchmark.Close();
}

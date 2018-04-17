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

#ifndef MACE_CORE_MACROS_H_
#define MACE_CORE_MACROS_H_

// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
#if defined(COMPILER_GCC3)
#define MACE_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define MACE_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define MACE_PREDICT_FALSE(x) (x)
#define MACE_PREDICT_TRUE(x) (x)
#endif

#define MACE_UNUSED(var) (void)(var)

#endif  // MACE_CORE_MACROS_H_

//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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

#endif  // MACE_CORE_MACROS_H_

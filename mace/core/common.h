//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_COMMON_H_
#define MACE_CORE_COMMON_H_

#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "mace/core/logging.h"

using std::set;
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

typedef int64_t index_t;

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)                              \
private:                                                                       \
  classname(const classname&) = delete;                                        \
  classname& operator=(const classname&) = delete
#endif

#define MACE_NOT_IMPLEMENTED MACE_CHECK(false, "not implemented")

// TODO: need to fine tune this
#define kCostPerGroup 1024000000

#endif // MACE_CORE_COMMON_H_

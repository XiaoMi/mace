//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_UTILS_H_
#define MACE_UTILS_UTILS_H_
namespace mace {
template <typename Integer>
Integer RoundUp(Integer i, Integer factor) {
  return (i + factor - 1) / factor * factor;
}

template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}
} //  namespace mace
#endif //  MACE_UTILS_UTILS_H_

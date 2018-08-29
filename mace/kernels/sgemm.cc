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

#include <algorithm>
#include <cstring>
#include <vector>

#include "mace/kernels/sgemm.h"

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

namespace mace {
namespace kernels {

void SGemm::operator()(const MatrixMap<float> &lhs,
                       const MatrixMap<float> &rhs,
                       MatrixMap<float> *result) {
  PackedBlock<float> packed_lhs;
  PackLhs(lhs, &packed_lhs);

  PackedBlock<float> packed_rhs;
  PackRhs(rhs, &packed_rhs);

  PackedBlock<float> packed_result;
  operator()(packed_lhs,
             packed_rhs,
             lhs.row(),
             lhs.col(),
             rhs.col(),
             &packed_result);
  UnPack(packed_result, result);
}

void SGemm::operator()(const PackedBlock<float> &lhs,
                       const PackedBlock<float> &rhs,
                       const index_t height,
                       const index_t depth,
                       const index_t width,
                       PackedBlock<float> *result) {
  (void) lhs;
  (void) rhs;
  (void) result;
  (void) height;
  (void) depth;
  (void) width;

  // (8, 8) * (8, 4)

  // (4, 4) * (4, 4)

  // remain
}

void SGemm::PackLhs(const MatrixMap<float> &lhs,
                    PackedBlock<float> *packed_block) {
  Pack(lhs, PackOrder::ColMajor, packed_block);
}

void SGemm::PackRhs(const MatrixMap<float> &rhs,
                    PackedBlock<float> *packed_block) {
  Pack(rhs, PackOrder::RowMajor, packed_block);
}

void SGemm::UnPack(const PackedBlock<float> &packed_result,
                   MatrixMap<float> *matrix_map) {
  (void) packed_result;
  (void) matrix_map;
}

void SGemm::Pack(const MatrixMap<float> &src,
                 const PackOrder order,
                 PackedBlock<float> *packed_block) {
  (void) src;
  (void) order;
  (void) packed_block;
}

}  // namespace kernels
}  // namespace mace

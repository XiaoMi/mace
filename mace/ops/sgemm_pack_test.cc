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

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <vector>

#include "mace/ops/sgemm.h"

namespace mace {
namespace ops {
namespace test {

namespace {
void TestPack(const std::vector<float> &data,
              const std::vector<float> &expected_data,
              const index_t height,
              const index_t width,
              Major src_order,
              PackOrder pack_order) {
  SGemm sg;
  SGemmMatrixMap<const float>
      src_matrix(1, height, width, src_order, data.data());
  PackedBlock packed;
  packed.Resize({height, width});
  if (pack_order == PackOrder::SGemmColMajor) {
    sg.PackLhs(src_matrix, &packed);
  } else {
    sg.PackRhs(src_matrix, &packed);
  }

  auto packed_data = packed.data<float>();
  for (index_t i = 0; i < packed.size(); ++i) {
    EXPECT_EQ(expected_data[i], packed_data[i]);
  }
}

void TestUnPack(const index_t height,
                const index_t width,
                Major src_order,
                PackOrder pack_order) {
  static auto seed = static_cast<unsigned int>(time(nullptr));
  const index_t matrix_size = height * width;
  std::vector<float> data(matrix_size);
  for (int i = 0; i < matrix_size; ++i) {
    data[i] = rand_r(&seed);
  }

  SGemmMatrixMap<const float>
      src_matrix(1, height, width, src_order, data.data());
  PackedBlock packed;
  packed.Resize({height, width});
  SGemm sg;
  if (pack_order == PackOrder::SGemmColMajor) {
    sg.PackLhs(src_matrix, &packed);
  } else {
    sg.PackRhs(src_matrix, &packed);
  }

  std::vector<float> unpacked(matrix_size);
  SGemmMatrixMap<float>
      unpacked_matrix(1, height, width, src_order, unpacked.data());
  sg.UnPack(packed, &unpacked_matrix);
  auto unpacked_data = unpacked.data();
  for (index_t i = 0; i < packed.size(); ++i) {
    EXPECT_EQ(data[i], unpacked_data[i]);
  }
}
}  // namespace


TEST(SGemmPackTest, Pack) {
  std::vector<float> data =
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};

  // For no-transpose lhs
  TestPack(data,
           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
           3, 4, Major::SGemmRowMajor, PackOrder::SGemmColMajor);
#if defined(MACE_ENABLE_NEON)
  TestPack(data,
           {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16},
           4, 4, Major::SGemmRowMajor, PackOrder::SGemmColMajor);
  TestPack(data,
           {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16, 17, 18, 19,
            20},
           5, 4, Major::SGemmRowMajor, PackOrder::SGemmColMajor);
#if defined(__aarch64__)
  TestPack(data,
           {1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11,
            15, 19, 23, 27, 31, 4, 8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36},
           9, 4, Major::SGemmRowMajor, PackOrder::SGemmColMajor);
#endif
#endif
  // For transpose-needed lhs
  TestPack(data,
           {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12},
           3, 4, Major::SGemmColMajor, PackOrder::SGemmColMajor);
#if defined(MACE_ENABLE_NEON)
  TestPack(data,
           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           4, 4, Major::SGemmColMajor, PackOrder::SGemmColMajor);
  TestPack(data,
           {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 5, 10, 15,
            20},
           5, 4, Major::SGemmColMajor, PackOrder::SGemmColMajor);
#if defined(__aarch64__)
  TestPack(data,
           {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
            22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 9, 18, 27, 36},
           9, 4, Major::SGemmColMajor, PackOrder::SGemmColMajor);
#endif
#endif
  // For no-transpose rhs
  TestPack(data,
           {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12},
           4, 3, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
#if defined(MACE_ENABLE_NEON)
  TestPack(data,
           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
           4, 4, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
  TestPack(data,
           {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 5, 10, 15,
            20},
           4, 5, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
#endif
  // For transpose-needed rhs
  TestPack(data,
           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
           4, 3, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
#if defined(MACE_ENABLE_NEON)
  TestPack(data,
           {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16},
           4, 4, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
  TestPack(data,
           {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16, 17, 18, 19,
            20},
           4, 5, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
#endif
}

TEST(SGemmPackTest, UnPack) {
  TestUnPack(4, 3, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 4, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 5, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 100, Major::SGemmRowMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 3, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 4, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 5, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
  TestUnPack(4, 100, Major::SGemmColMajor, PackOrder::SGemmRowMajor);
}

}  // namespace test
}  // namespace ops
}  // namespace mace


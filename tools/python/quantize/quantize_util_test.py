# Copyright 2019 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import quantize.quantize_util


class TestQuantize(unittest.TestCase):

    def test_quantize_dequantize(self):
        test_input = np.random.rand(20, 30) * 5
        quantized_data = quantize_util.quantize(test_input)
        dequantized_output = quantize_util.dequantize(quantized_data)
        np.testing.assert_array_almost_equal(test_input, dequantized_output, 2)


if __name__ == '__main__':
    unittest.main()

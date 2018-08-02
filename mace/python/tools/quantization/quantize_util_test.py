import unittest
import numpy as np
import quantize_util


class TestQuantize(unittest.TestCase):

    def test_quantize_dequantize(self):
        test_input = np.random.rand(20, 30) * 5
        quantized_data = quantize_util.quantize(test_input)
        dequantized_output = quantize_util.dequantize(quantized_data)
        np.testing.assert_array_almost_equal(test_input, dequantized_output, 2)


if __name__ == '__main__':
    unittest.main()

import numpy as np
import math


class QuantizedData(object):
    def __init__(self):
        self._data = None
        self._scale = 0
        self._zero = 0

    @property
    def data(self):
        return self._data

    @property
    def scale(self):
        return self._scale

    @property
    def zero(self):
        return self._zero

    @data.setter
    def data(self, data):
        self._data = data

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @zero.setter
    def zero(self, zero):
        self._zero = zero


def adjust_range(in_min, in_max, non_zero):
    out_max = max(0.0, in_max)
    out_min = min(0.0, in_min)
    if non_zero:
        out_min = min(out_min, in_min - (out_max - in_min) / 254.0)
    scale = (out_max - out_min) / 255.0
    eps = 1e-6
    if out_min < -eps and out_max > eps:
        zero = -out_min / scale
        zero_int = int(round(zero))
        if abs(zero - zero_int) > eps:
            if zero < zero_int or non_zero:
                zero_int = int(math.ceil(zero))
                scale = out_max / (255.0 - zero_int)
            else:
                scale = -out_min / zero_int
    elif out_min > -eps:
        zero_int = 0
    else:
        zero_int = 255

    return scale, zero_int


def cal_multiplier_and_shift(scale):
    """
    In order to use gemmlowp, we need to use gemmlowp-like transform
    :param scale:
    :return: multiplier, shift
    """
    assert scale > 0, "scale should > 0, but get %s" % scale
    assert scale < 1, "scale should < 1, but get %s" % scale
    multiplier = scale
    s = 0
    # make range [1/2, 1)
    while multiplier < 0.5:
        multiplier *= 2.0
        s += 1
    # convert scale to fixed-point
    q = int(round(multiplier * (1 << 31)))
    assert q <= (1 << 31)
    if q == (1 << 31):
        q /= 2
        s -= 1
    assert s >= 0
    return q, s


def quantize_with_scale_and_zero(data, scale, zero):
    output = np.round(zero + data / scale).astype(int)
    quantized_data = QuantizedData()
    quantized_data.data = output
    quantized_data.scale = scale
    quantized_data.zero = zero
    return quantized_data


def quantize(data):
    np_data = np.array(data).astype(float)
    in_min = np_data.min()
    in_max = np_data.max()
    scale, zero = adjust_range(in_min, in_max, non_zero=True)
    output = np.clip((np.round(zero + data / scale).astype(int)), 0, 255)

    quantized_data = QuantizedData()
    quantized_data.data = output
    quantized_data.scale = scale
    quantized_data.zero = zero
    return quantized_data


def dequantize(quantized_data):
    return quantized_data.scale * (quantized_data.data - quantized_data.zero)

import numpy as np
import math

from mace.python.tools.converter_tool.base_converter import DeviceType


class QuantizedData(object):
    def __init__(self):
        self._data = None
        self._scale = 0
        self._zero = 0
        self._minval = 0.0
        self._maxval = 0.0

    @property
    def data(self):
        return self._data

    @property
    def scale(self):
        return self._scale

    @property
    def zero(self):
        return self._zero

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    @data.setter
    def data(self, data):
        self._data = data

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @zero.setter
    def zero(self, zero):
        self._zero = zero

    @minval.setter
    def minval(self, minval):
        self._minval = minval

    @maxval.setter
    def maxval(self, maxval):
        self._maxval = maxval


def adjust_range(in_min, in_max, device, non_zero):
    if device in [DeviceType.HEXAGON.value, DeviceType.HTA.value]:
        return adjust_range_for_hexagon(in_min, in_max)

    out_max = max(0.0, in_max)
    out_min = min(0.0, in_min)
    if non_zero:
        out_min = min(out_min, in_min - (out_max - in_min) / 254.0)
    scale = (out_max - out_min) / 255.0
    eps = 1e-6
    if out_min < -eps and out_max > eps:
        zero = -out_min / scale
        zero_int = int(round(zero))
        if abs(zero - zero_int) > eps and non_zero:
            zero_int = int(math.ceil(zero))
    elif out_min > -eps:
        zero_int = 0
    else:
        zero_int = 255

    return scale, zero_int, -zero_int*scale, (255-zero_int)*scale


def adjust_range_for_hexagon(in_min, in_max):
    out_max = max(0.0, in_max)
    out_min = min(0.0, in_min)
    scale = (out_max - out_min) / 255.0
    eps = 1e-6
    if out_min < -eps and out_max > eps:
        zero = -out_min / scale
        zero_int = int(round(zero))
        # if zero_int <=0 or >= 255, try to avoid divide by 0,
        # else, try to make adjustment as small as possible
        ceil = int(math.ceil(zero))
        keep_max = (ceil - zero) / out_max < (zero + 1 - ceil) / -out_min
        if zero_int <= 0 or (zero_int < 254 and keep_max):
            zero_int = ceil
            scale = out_max / (255.0 - zero_int)
        else:
            scale = -out_min / zero_int
    elif out_min > -eps:
        zero_int = 0
    else:
        zero_int = 255

    return scale, zero_int, -zero_int*scale, (255-zero_int)*scale


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
    output = np.round(zero + data / scale).astype(np.int32)
    quantized_data = QuantizedData()
    quantized_data.data = output
    quantized_data.scale = scale
    quantized_data.zero = zero
    return quantized_data


def quantize(data, device, non_zero):
    np_data = np.array(data).astype(float)
    in_min = np_data.min()
    in_max = np_data.max()
    scale, zero, out_min, out_max = adjust_range(in_min, in_max, device,
                                                 non_zero=non_zero)
    output = np.clip((np.round(zero + data / scale).astype(np.int32)), 0, 255)

    quantized_data = QuantizedData()
    quantized_data.data = output
    quantized_data.scale = scale
    quantized_data.zero = zero
    quantized_data.minval = out_min
    quantized_data.maxval = out_max
    return quantized_data


def quantize_bias_for_hexagon(data):
    np_data = np.array(data).astype(float)
    max_val = max(abs(np_data.min()), abs(np_data.max()))
    in_min = -max_val
    in_max = max_val
    scale = (in_max - in_min) / 2**32
    zero = 0
    output = np.clip((np.round(zero + data / scale).astype(np.int64)),
                     -2**31, 2**31 - 1)

    quantized_data = QuantizedData()
    quantized_data.data = output
    quantized_data.scale = scale
    quantized_data.zero = zero
    quantized_data.minval = in_min
    quantized_data.maxval = in_max
    return quantized_data


def dequantize(quantized_data):
    return quantized_data.scale * (quantized_data.data - quantized_data.zero)

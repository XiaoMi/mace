import argparse
import numpy as np


class QuantizeStat(object):
    def __init__(self):
        pass

    @staticmethod
    def run(log_file, percentile, enhance, enhance_ratio):
        res = {}
        tensor_ranges = {}
        with open(log_file) as log:
            for line in log:
                if line.find("Tensor range @@") != -1:
                    tensor_name, minmax = line.split("@@")[1:]
                    min_val, max_val = [float(i) for i in
                                        minmax.strip().split(",")]
                    if tensor_name not in tensor_ranges:
                        tensor_ranges[tensor_name] = ([], [])
                    tensor_ranges[tensor_name][0].append(min_val)
                    tensor_ranges[tensor_name][1].append(max_val)

        for tensor_name in tensor_ranges:
            samples = len(tensor_ranges[tensor_name][0])
            tensor_min = np.percentile(tensor_ranges[tensor_name][0],
                                       percentile)
            tensor_max = np.percentile(tensor_ranges[tensor_name][1],
                                       100 - percentile)
            assert tensor_min < tensor_max
            if not enhance or samples <= 1:
                res[tensor_name] = (tensor_min, tensor_max)
            else:
                """
                Enhancement mode:
                This policy eliminates outliers that cause long-tail
                statistical range. We try to reduce as much range as it could
                while retaining more samples. d(range)/d(sample_quantile) is
                used to measure this qualitatively.
                """
                tensor_mins = np.sort(tensor_ranges[tensor_name][0])
                tensor_maxs = np.sort(tensor_ranges[tensor_name][1])[::-1]
                cur_min_idx = 0
                cur_max_idx = 0
                cur_min = tensor_min
                cur_max = tensor_max
                for i in xrange(samples):
                    if tensor_mins[i] + 0.1 > cur_max:
                        break

                    delta_range = (tensor_mins[i] - cur_min) / (cur_max - cur_min)  # noqa
                    delta_quantile = float(i - cur_min_idx) / (samples - cur_min_idx)  # noqa
                    if delta_quantile > 0 and delta_range / delta_quantile > enhance_ratio:  # noqa
                        cur_min_idx = i
                        cur_min = tensor_mins[i]

                    if cur_min + 0.1 > tensor_maxs[i]:
                        break

                    delta_range = (cur_max - tensor_maxs[i]) / (cur_max - cur_min)  # noqa
                    delta_quantile = float(i - cur_max_idx) / (samples - cur_max_idx)  # noqa
                    if delta_quantile > 0 and delta_range / delta_quantile > enhance_ratio:  # noqa
                        cur_max_idx = i
                        cur_max = tensor_maxs[i]

                res[tensor_name] = (cur_min, cur_max)

        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="path of log file that records tensor range")
    parser.add_argument(
        "--percentile",
        type=int,
        default=0,
        help="range percentile")
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="range percentile")
    parser.add_argument(
        "--enhance_ratio",
        type=int,
        default=10,
        help="enhance ratio")
    FLAGS, unparsed = parser.parse_known_args()

    res = QuantizeStat.run(FLAGS.log_file, FLAGS.percentile, FLAGS.enhance,
                           FLAGS.enhance_ratio)
    for tensor in res:
        print("%s@@%f,%f" % (tensor, res[tensor][0], res[tensor][1]))

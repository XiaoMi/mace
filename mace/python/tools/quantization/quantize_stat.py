import argparse
import numpy as np


class QuantizeStat(object):
    def __init__(self):
        pass

    @staticmethod
    def run(log_file, percentile):
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
            tensor_min = np.percentile(tensor_ranges[tensor_name][0],
                                       percentile)
            tensor_max = np.percentile(tensor_ranges[tensor_name][1],
                                       100 - percentile)
            assert tensor_min < tensor_max
            res[tensor_name] = (tensor_min, tensor_max)

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
        default=5,
        help="range percentile")
    FLAGS, unparsed = parser.parse_known_args()

    res = QuantizeStat.run(FLAGS.log_file, FLAGS.percentile)
    for tensor in res:
        print("%s@@%f,%f" % (tensor, res[tensor][0], res[tensor][1]))

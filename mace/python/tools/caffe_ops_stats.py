from mace.proto import caffe_pb2
import google.protobuf.text_format
import operator
import functools
import argparse
import sys
import six
import os.path

FLAGS = None

def main(unused_args):
  if not os.path.isfile(FLAGS.input):
    print 'input model file not exist'
    return -1
  net = caffe_pb2.NetParameter()
  with open(FLAGS.input) as f:
    google.protobuf.text_format.Merge(str(f.read()), net)

  ops = {}
  for layer in net.layer:
    if layer.type not in ops:
      ops[layer.type] = 1
    else:
      ops[layer.type] += 1

  for key, value in sorted(ops.items(), key=operator.itemgetter(1)):
    print key, ":", value

def parse_args():
  '''Parses command line arguments.'''
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input',
    type=str,
    default='',
    help='Caffe \'GraphDef\' file to load.')
  return parser.parse_known_args()

if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

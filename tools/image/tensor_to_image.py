import argparse
import os
import sys

import six

import numpy as np
import tensorflow as tf

# TODO(liyin): use dataset api and estimator with distributed strategy

FLAGS = None


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="tensor file/dir path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="image output dir.")
    parser.add_argument(
        "--image_shape",
        type=str,
        help="target image shape, e.g, 224,224,3")
    return parser.parse_known_args()


def tensors_to_images(input_files, image_shape):
    with tf.Graph().as_default():
        input = tf.placeholder(tf.float32, shape=image_shape, name='input')
        output = tf.placeholder(tf.string, name='output_file')
        # use the second channel if it is gray image
        if image_shape[2] == 2:
            _, input = tf.split(input, 2, axis=2)
        tensor_data = tf.image.convert_image_dtype(input,
                                                   tf.uint8,
                                                   saturate=True)
        image_data = tf.image.encode_jpeg(tensor_data, quality=100)
        writer = tf.write_file(output, image_data, name='output_writer')

        with tf.Session() as sess:
            for i in xrange(len(input_files)):
                input_data = np.fromfile(input_files[i], dtype=np.float32) \
                    .reshape(image_shape)
                output_file = os.path.join(FLAGS.output_dir, os.path.splitext(
                    os.path.basename(input_files[i]))[0] + '.jpg')
                sess.run(writer, feed_dict={'input:0': input_data,
                                            'output_file:0': output_file})


def main(unused_args):
    if not os.path.exists(FLAGS.input):
        print("input does not exist: %s" % FLAGS.input)
        sys.exit(-1)

    input_files = []
    if os.path.isdir(FLAGS.input):
        input_files.extend([os.path.join(FLAGS.input, tensor)
                            for tensor in os.listdir(FLAGS.input)])
    else:
        input_files.append(FLAGS.input)

    image_shape = [int(dim) for dim in FLAGS.image_shape.split(',')]
    tensors_to_images(input_files, image_shape)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)

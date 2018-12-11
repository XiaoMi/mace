import argparse
import os
import sys

import six

import tensorflow as tf

# TODO(liyin): use dataset api and estimator with distributed strategy

FLAGS = None


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="image file/dir path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="tensor output dir.")
    parser.add_argument(
        "--image_shape",
        type=str,
        help="target image shape, e.g, 224,224,3")
    parser.add_argument(
        "--mean",
        type=str,
        default="",
        help="rgb mean value that should subtract from image value,"
             " e.g, 128,128,128.")
    return parser.parse_known_args()


def images_to_tensors(input_files, image_shape, mean_values=None):
    with tf.Graph().as_default():
        image_data = tf.placeholder(tf.string, name='input')
        image_data = tf.image.decode_image(image_data,
                                           channels=image_shape[2])
        if mean_values:
            image_data = tf.cast(image_data, dtype=tf.float32)
            mean_tensor = tf.constant(mean_values, dtype=tf.float32,
                                      shape=[1, 1, image_shape[2]])
            image_data = (image_data - mean_tensor) / 255.0
        else:
            image_data = tf.image.convert_image_dtype(image_data,
                                                      dtype=tf.float32)
            image_data = tf.subtract(image_data, 0.5)
            image_data = tf.multiply(image_data, 2.0)

        image_data = tf.expand_dims(image_data, 0)
        image_data = tf.image.resize_bilinear(image_data,
                                              image_shape[:2],
                                              align_corners=False)

        with tf.Session() as sess:
            for i in xrange(len(input_files)):
                with tf.gfile.FastGFile(input_files[i], 'rb') as f:
                    src_image = f.read()
                    dst_image = sess.run(image_data,
                                         feed_dict={'input:0': src_image})
                    output_file = os.path.join(FLAGS.output_dir,
                                               os.path.splitext(
                                                   os.path.basename(
                                                       input_files[i]))[
                                                   0] + '.dat')
                    dst_image.tofile(output_file)


def main(unused_args):
    if not os.path.exists(FLAGS.input):
        print("input does not exist: %s" % FLAGS.input)
        sys.exit(-1)

    input_files = []
    if os.path.isdir(FLAGS.input):
        input_files.extend([os.path.join(FLAGS.input, image)
                            for image in os.listdir(FLAGS.input)])
    else:
        input_files.append(FLAGS.input)

    image_shape = [int(dim) for dim in FLAGS.image_shape.split(',')]
    mean_values = None
    if FLAGS.mean:
        mean_values = [float(mean) for mean in FLAGS.mean.split(',')]
    images_to_tensors(input_files, image_shape, mean_values=mean_values)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)

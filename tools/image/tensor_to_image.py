import argparse
import os
import sys
import numpy as np
import tensorflow as tf

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
    for i in xrange(len(input_files)):
        with tf.Session() as sess:
            tensor_data = np.fromfile(input_files[i], dtype=np.float32) \
                .reshape(image_shape)
            # use the second channel if it is gray image
            if image_shape[2] == 2:
                _, tensor_data = tf.split(tensor_data, 2, axis=2)
            tensor_data = tf.image.convert_image_dtype(tensor_data,
                                                       tf.uint8,
                                                       saturate=True)
            image_data = tf.image.encode_jpeg(tensor_data, quality=100)
            image = sess.run(image_data)
            output_file = os.path.join(FLAGS.output_dir, os.path.splitext(
                os.path.basename(input_files[i]))[0] + '.jpg')
            writer = tf.write_file(output_file, image)
            sess.run(writer)


def main(unused_args):
    if not os.path.exists(FLAGS.input):
        print ("input does not exist: %s" % FLAGS.input)
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

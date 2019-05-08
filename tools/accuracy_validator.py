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

import os.path
import numpy as np
from PIL import Image


class AccuracyValidator(object):
    """Accuracy Validator Plugin:
    Usage: This script is used to calculate the accuracy(like Top-1)
           of MACE model.
           User could reload this validator script to do
           other accuracy validation(like MIOU for segmentation),
           the new script's interface should be same
           with current AccuracyValidator exactly,

    Warning: Do not use relative path in this script.
    """
    def __init__(self, **kwargs):
        # absolute path
        validation_set_image_dir = \
            '/path/to/your/validation/set/directory'
        validation_set_label_file_path =\
            '/path/to/imagenet_groundtruth_labels.txt'
        black_list_file_path = \
            '/path/to/imagenet_blacklist.txt'
        imagenet_classes_file = \
            '/path/to/imagenet_classes.txt'
        self._imagenet_classes = [
            line.rstrip('\n') for line in open(imagenet_classes_file)]
        imagenet_classes_map = {}
        for idx in range(len(self._imagenet_classes)):
            imagenet_classes_map[self._imagenet_classes[idx]] = idx
        black_list = [
            int(line.rstrip('\n')) for line in open(black_list_file_path)]

        self._samples = []
        self._labels = [0]  # image id start from 1
        self._correct_count = 0

        for img_file in os.listdir(validation_set_image_dir):
            if img_file.endswith(".JPEG"):
                img_id = int(os.path.splitext(img_file)[0].split('_')[-1])
                if img_id not in black_list:
                    self._samples.append(
                        os.path.join(validation_set_image_dir, img_file))
        for label in open(validation_set_label_file_path):
            label = label.rstrip('\n')
            self._labels.append(imagenet_classes_map[label])

    def sample_size(self):
        """
        :return: the size of samples in validation set
        """
        return len(self._samples)

    def batch_size(self):
        """
        batch size to do validation to speed up validation.
        Keep same with batch size of input_shapes
         in model deployment file(.yml). do not set too large
        :return: batch size
        """
        return 1

    def preprocess(self, sample_idx_start, sample_idx_end, **kwargs):
        """
        pre-process the input sample
        :param sample_idx_start: start index of the sample.
        :param sample_idx_end: end index of the sample(not include).
        :param kwargs: other parameters.
        :return: the batched inputs' map(name: data) feed into your model
        """
        inputs = {}
        batch_sample_data = []
        sample_idx_end = min(sample_idx_end, self.sample_size())
        for sample_idx in range(sample_idx_start, sample_idx_end):
            sample_file_path = self._samples[sample_idx]
            sample_img = Image.open(sample_file_path).resize((224, 224))
            sample_data = np.asarray(sample_img, dtype=np.float32)
            sample_data = (2.0 / 255.0) * sample_data - 1.0
            batch_sample_data.append(sample_data.tolist())
        inputs["input"] = batch_sample_data
        return inputs

    def postprocess(self,
                    sample_idx_start,
                    sample_idx_end,
                    output_map,
                    **kwargs):
        """
        post-process the outputs of your model and calculate the accuracy
        :param sample_idx_start: start index of input sample
        :param sample_idx_end: end index of input sample
        :param output_map: output map of the model
        :param kwargs: other parameters.
        :return: None
        """
        output = output_map['MobilenetV2/Predictions/Reshape_1']
        sample_idx_end = min(sample_idx_end, self.sample_size())
        batch_size = sample_idx_end - sample_idx_start
        output = np.array(output).reshape((batch_size, -1))
        output = np.argmax(output, axis=-1)
        output_idx = 0
        for sample_idx in range(sample_idx_start, sample_idx_end):
            sample_file_path = self._samples[sample_idx]
            img_id = int(os.path.splitext(sample_file_path)[0].split('_')[-1])
            if output[output_idx] == self._labels[img_id]:
                self._correct_count += 1
            else:
                print(img_id, 'predict %s vs gt %s' %
                      (self._imagenet_classes[output[output_idx]],
                       self._imagenet_classes[self._labels[img_id]]))
            output_idx += 1

    def result(self):
        """
        print or show the result
        :return: None
        """
        print("==========================================")
        print("Top 1 accuracy: %f" %
              (self._correct_count * 1.0 / self.sample_size()))
        print("==========================================")


if __name__ == '__main__':
    # sample usage code
    validator = AccuracyValidator()
    sample_size = validator.sample_size()
    val_batch_size = validator.batch_size()
    for i in range(0, sample_size, val_batch_size):
        inputs = validator.preprocess(i, i+val_batch_size)
        print(np.array(inputs['input']).shape)

        output_map = {
            'MobilenetV2/Predictions/Reshape_1': np.array([[0, 1], [1, 0]])
        }
        validator.postprocess(i, i+val_batch_size, output_map)

    validator.result()

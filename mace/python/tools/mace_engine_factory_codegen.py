# Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

import argparse

from jinja2 import Environment, FileSystemLoader


FLAGS = None


def gen_mace_engine_factory(model_tags, template_dir, model_type,
                            embed_model_data, output_dir):
    # Create the jinja2 environment.
    j2_env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True)
    # generate mace_run BUILD file
    template_name = 'mace_engine_factory.h.jinja2'
    source = j2_env.get_template(template_name).render(
        model_tags=model_tags,
        embed_model_data=embed_model_data,
        model_type=model_type,
    )
    with open(output_dir + '/mace_engine_factory.h', "wb") as f:
        f.write(source)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_tag",
        type=str,
        default="",
        help="model tag")
    parser.add_argument(
        "--template_dir", type=str, default="", help="template path")
    parser.add_argument(
        "--output_dir", type=str, default="", help="output path")
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="[source|pb] model load type")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    gen_mace_engine_creator(FLAGS.model_tag, FLAGS.template_dir,
                            FLAGS.model_type, FLAGS.output_dir)

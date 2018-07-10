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

import sys


def is_static_lib(lib_name):
    return lib_name.endswith('.a') or lib_name.endswith('.lo')


def merge_libs(input_libs,
               output_lib_path,
               mri_script):
    # make static library
    mri_stream = ""
    mri_stream += "create %s\n" % output_lib_path
    for lib in input_libs:
        if is_static_lib(lib):
            mri_stream += ("addlib %s\n" % lib)
    mri_stream += "save\n"
    mri_stream += "end\n"
    with open(mri_script, 'w') as tmp:
        tmp.write(mri_stream)


if __name__ == '__main__':
    merge_libs(sys.argv[1:-2], sys.argv[-2], sys.argv[-1])

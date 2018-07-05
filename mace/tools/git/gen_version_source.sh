#!/usr/bin/env bash
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

OUTPUT_FILENAME=$1
if [[ -z "${OUTPUT_FILENAME}}" ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

DATE_STR=$(date +%Y%m%d)
GIT_VERSION=$(git describe --long --tags)
if [[ $? != 0 ]]; then
  GIT_VERSION=unknown-${DATE_STR}
else
  GIT_VERSION=${GIT_VERSION}-${DATE_STR}
fi

cat <<EOF > ${OUTPUT_FILENAME}
// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is a generated file. DO NOT EDIT!

namespace mace {
__attribute__((visibility ("default")))
const char *MaceVersion() { return "MACEVER-${GIT_VERSION}" + 8; }
}  // namespace mace
EOF

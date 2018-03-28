#!/usr/bin/env bash
#
# Copyright (c) 2017 XiaoMi All rights reserved.
#

OUTPUT_FILENAME=$1
if [[ -z "${OUTPUT_FILENAME}}" ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

DATE_STR=$(date +%Y%m%d)
GIT_VERSION=$(git describe --long --tags)
if [[ $? != 0 ]]; then
  GIT_VERSION=unknown
else
  GIT_VERSION=${GIT_VERSION}-${DATE_STR}
fi

cat <<EOF > ${OUTPUT_FILENAME}
//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// This is a generated file, DO NOT EDIT

namespace mace {
  const char *MaceVersion() { return "${GIT_VERSION}"; }
}  // namespace mace
EOF

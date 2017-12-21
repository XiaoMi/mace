#!/usr/bin/env bash
#
# Copyright (c) 2017 XiaoMi All rights reserved.
#

OUTPUT_FILENAME=$1
if [[ -z "${OUTPUT_FILENAME}}" ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

GIT_VERSION=$(git describe --long --tags)
if [[ $? != 0 ]]; then
  GIT_VERSION=unknown
fi

cat <<EOF > ${OUTPUT_FILENAME}
#include "mace/core/public/version.h"
const char *MaceVersion() { return MACE_VERSION_STRING; }
const char *MaceGitVersion() { return "${GIT_VERSION}"; }
EOF

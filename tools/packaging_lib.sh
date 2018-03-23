#!/bin/bash

Usage() {
  echo "Usage: bash tools/packaging_lib.sh libmace_output_dir"
}

if [ $# -lt 1 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

LIBMACE_BUILD_DIR=$1

pushd $LIBMACE_BUILD_DIR/$PROJECT_NAME
ls | grep -v build | xargs tar cvzf libmace_${PROJECT_NAME}.tar.gz
popd

echo "Packaging done!"

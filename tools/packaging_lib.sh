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

TAR_PACKAGE_NAME=libmace_${PROJECT_NAME}.tar.gz

pushd $LIBMACE_BUILD_DIR/$PROJECT_NAME
if [ -f $TAR_PACKAGE_NAME ]; then
  rm -f $TAR_PACKAGE_NAME
fi
ls | grep -v build | xargs tar cvzf $TAR_PACKAGE_NAME
popd

echo "Packaging done!"

#!/bin/bash

Usage() {
  echo "Usage: bash tools/download_and_link_lib.sh libmace_v7_dsp"
}

if [ $# -lt 1 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

LIB_FOLDER_NAME=$1

if [ ! -d "${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}" ]; then
  wget -P ${LIBMACE_SOURCE_DIR}/lib http://cnbj1-inner-fds.api.xiaomi.net/libmace/libs/${LIBMACE_TAG}/${LIB_FOLDER_NAME}.tar.gz && \
    tar xvzf ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}.tar.gz -C ${LIBMACE_SOURCE_DIR}/lib/ || exit 1
  echo "${LIB_FOLDER_NAME} download successfully!"
else
  echo "${LIB_FOLDER_NAME} already exists!"
fi

echo "Create link 'mace' of downloaded or existed ${LIB_FOLDER_NAME}"
if [ -L ${LIBMACE_SOURCE_DIR}/lib/mace ]; then
  unlink ${LIBMACE_SOURCE_DIR}/lib/mace
fi
ln -s ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME} ${LIBMACE_SOURCE_DIR}/lib/mace && \
  rm -rf ${LIBMACE_SOURCE_DIR}/lib/${LIB_FOLDER_NAME}.tar.gz || exit 1

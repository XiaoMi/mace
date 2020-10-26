#!/bin/bash

BUILD_DIR=build/micro/host
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

cmake ../../../micro \
  -DCMAKE_INSTALL_PREFIX=install \
  $@ || exit 1

cmake --build . --target install -- -j || exit 1

cd ../../..

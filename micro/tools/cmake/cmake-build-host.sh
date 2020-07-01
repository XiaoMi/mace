#!/bin/bash
BUILD_DIR=build/cmake-build/host
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

cmake ../../.. \
  -DMACE_MICRO_ENABLE_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=install \
  $@ || exit 1

cmake --build . -- -j || exit 1

cd ../../..

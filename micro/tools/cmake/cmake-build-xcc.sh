#!/bin/bash

BUILD_DIR=build/micro/xcc

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../../micro \
  -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain/xcc.cmake \
  -DMACE_MICRO_ENABLE_XTENSA=ON \
  -DCMAKE_INSTALL_PREFIX=install \
  -DMACE_MICRO_ENABLE_EXAMPLES=OFF \
  -DMICRO_MODEL_NAME=kws_tc_resnet8 \
  -DMICRO_DATA_NAME=kws \
  $@ || exit 1

cmake --build . --target install -- -j || exit 1

cd ../../..

#!/bin/bash

set -e

LIB_DIR=builds/lib
INCLUDE_DIR=builds/include/mace/public

mkdir -p $LIB_DIR
mkdir -p $INCLUDE_DIR

# generate version code
rm -rf mace/codegen/version
mkdir -p mace/codegen/version
bash mace/tools/git/gen_version_source.sh mace/codegen/version/version.cc

# generate tuning code
rm -rf mace/codegen/tuning
mkdir -p mace/codegen/tuning
python mace/python/tools/binary_codegen.py --output_path=mace/codegen/tuning/tuning_params.cc

# copy include headers
cp mace/public/*.h $INCLUDE_DIR/

echo "build lib for armeabi-v7a"
rm -rf $LIB_DIR/armeabi-v7a
mkdir -p $LIB_DIR/armeabi-v7a
bazel build --config android --config optimization mace:libmace --cpu=armeabi-v7a --define neon=true --define openmp=true
cp bazel-bin/mace/libmace.so $LIB_DIR/armeabi-v7a/

echo "build lib for armeabi-v7a with hexagon dsp"
rm -rf $LIB_DIR/armeabi-v7a/hexagon-dsp
mkdir -p $LIB_DIR/armeabi-v7a/hexagon-dsp
bazel build --config android --config optimization mace:libmace --cpu=armeabi-v7a --define neon=true --define openmp=true --define hexagon=true
cp bazel-bin/mace/libmace.so $LIB_DIR/armeabi-v7a/hexagon-dsp/
cp third_party/nnlib/*so $LIB_DIR/armeabi-v7a/hexagon-dsp/

echo "build lib for arm64-v8a"
rm -rf $LIB_DIR/arm64-v8a
mkdir -p $LIB_DIR/arm64-v8a
bazel build --config android --config optimization mace:libmace --cpu=arm64-v8a --define neon=true --define openmp=true
cp bazel-bin/mace/libmace.so $LIB_DIR/arm64-v8a/

echo "build lib for linux-x86-64"
rm -rf $LIB_DIR/linux-x86-64
mkdir -p $LIB_DIR/linux-x86-64
bazel build --config optimization mace:libmace --define openmp=true
cp bazel-bin/mace/libmace.so $LIB_DIR/linux-x86-64/


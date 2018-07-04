#!/bin/bash

set -e

LIB_DIR=build/lib
INCLUDE_DIR=build/include/mace/public

mkdir -p $LIB_DIR
mkdir -p $INCLUDE_DIR

# copy include headers
cp mace/public/*.h $INCLUDE_DIR/

echo "build lib for armeabi-v7a"
mkdir -p $LIB_DIR/armeabi-v7a
rm -f $LIB_DIR/armeabi-v7a/*
bazel build --config android mace:libmace --define neon=true --define openmp=true --define hexagon=true --cpu=armeabi-v7a
cp bazel-bin/mace/libmace.so $LIB_DIR/armeabi-v7a/
cp third_party/nnlib/*so $LIB_DIR/armeabi-v7a/

echo "build lib for arm64-v8a"
mkdir -p $LIB_DIR/arm64-v8a
rm -f $LIB_DIR/arm64-v8a/*
bazel build --config android mace:libmace --define neon=true --define openmp=true --cpu=arm64-v8a
cp bazel-bin/mace/libmace.so $LIB_DIR/arm64-v8a/

echo "build lib for linux-x86-64"
mkdir -p $LIB_DIR/linux-x86-64
rm -f $LIB_DIR/linux-x86-64/*
bazel build mace:libmace --define openmp=true
cp bazel-bin/mace/libmace.so $LIB_DIR/linux-x86-64/






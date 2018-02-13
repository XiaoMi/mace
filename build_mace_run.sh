#!/bin/bash

Usage() {
  echo "Usage: bash tools/build_mace_run.sh production_mode model_output_dir hexagon_mode"
}

if [ $# -lt 3 ]; then
  Usage
  exit 1
fi

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

PRODUCTION_MODE=$1
MODEL_OUTPUT_DIR=$2
HEXAGON_MODE=$3

if [ "$PRODUCTION_MODE" = 1 ]; then
  PRODUCTION_MODE_BUILD_FLAGS="--define production=true"
fi

if [ x"$TARGET_ABI" = x"host" ]; then
  bazel build --verbose_failures -c opt --strip always codegen:generated_models \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    $PRODUCTION_MODE_BUILD_FLAGS || exit 1

  bazel build --verbose_failures -c opt --strip always examples:mace_run \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    $PRODUCTION_MODE_BUILD_FLAGS || exit 1
else
  if [ "$HEXAGON_MODE" = 1 ]; then
    HEXAGON_MODE_BUILD_FLAG="--define hexagon=true"
  fi

  bazel build --verbose_failures -c opt --strip always examples:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=${TARGET_ABI} \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    $PRODUCTION_MODE_BUILD_FLAGS \
    $HEXAGON_MODE_BUILD_FLAG || exit 1
fi

if [ "$PRODUCTION_MODE" = 1 ]; then
  cp $GENERATED_MODEL_LIB_PATH $MODEL_OUTPUT_DIR/libmace_${MODEL_TAG}.a
fi

if [ -f "$MODEL_OUTPUT_DIR/mace_run" ]; then
  rm -rf $MODEL_OUTPUT_DIR/mace_run
fi
cp bazel-bin/examples/mace_run $MODEL_OUTPUT_DIR

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
  bazel build --verbose_failures -c opt --strip always //mace/codegen:generated_models \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    --copt="-O3" \
    $PRODUCTION_MODE_BUILD_FLAGS || exit 1

  bazel build --verbose_failures -c opt --strip always //mace/tools/validation:mace_run \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    --copt="-O3" \
    $PRODUCTION_MODE_BUILD_FLAGS || exit 1
else
  if [ "$HEXAGON_MODE" = 1 ]; then
    HEXAGON_MODE_BUILD_FLAG="--define hexagon=true"
  fi

  if [ x"$TARGET_ABI" = x"arm64-v8a" ]; then
    NEON_ENABLE_FLAG="--define neon=true"
  fi

  bazel build --verbose_failures -c opt --strip always //mace/tools/validation:mace_run \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=${TARGET_ABI} \
    --copt="-std=c++11" \
    --copt="-D_GLIBCXX_USE_C99_MATH_TR1" \
    --copt="-Werror=return-type" \
    --copt="-DMACE_OBFUSCATE_LITERALS" \
    --copt="-DMACE_MODEL_TAG=${MODEL_TAG}" \
    --define openmp=true \
    --define neon=true \
    --copt="-O3" \
    $NEON_ENABLE_FLAG \
    $PRODUCTION_MODE_BUILD_FLAGS \
    $HEXAGON_MODE_BUILD_FLAG || exit 1
fi

rm -rf $MODEL_OUTPUT_DIR/libmace_${MODEL_TAG}.a
cp $GENERATED_MODEL_LIB_PATH $MODEL_OUTPUT_DIR/libmace_${MODEL_TAG}.a

if [ -f "$MODEL_OUTPUT_DIR/mace_run" ]; then
  rm -rf $MODEL_OUTPUT_DIR/mace_run
fi
cp bazel-bin/mace/tools/validation/mace_run $MODEL_OUTPUT_DIR
if [ "$EMBED_MODEL_DATA" = 0 ]; then
  cp mace/codegen/models/${MODEL_TAG}/${MODEL_TAG}.data $MODEL_OUTPUT_DIR
fi

# copy model header file to build output dir
cp mace/codegen/models/${MODEL_TAG}/${MODEL_TAG}.h $MODEL_OUTPUT_DIR

#!/bin/bash

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

python mace/python/tools/encrypt_opencl_codegen.py \
    --cl_kernel_dir=./mace/kernels/opencl/cl/ \
    --output_path=${CODEGEN_DIR}/opencl/opencl_encrypt_program.cc || exit 1


rm -rf ${CODEGEN_DIR}/version
mkdir ${CODEGEN_DIR}/version
bash mace/tools/git/gen_version_source.sh ${CODEGEN_DIR}/version/version.cc || exit 1

#!/bin/bash

CL_KERNEL_DIR_TAG=$1

CURRENT_DIR=`dirname $0`
source ${CURRENT_DIR}/env.sh

rm -rf ${CODEGEN_DIR}/version
mkdir ${CODEGEN_DIR}/version
bash mace/tools/git/gen_version_source.sh ${CODEGEN_DIR}/version/version.cc || exit 1

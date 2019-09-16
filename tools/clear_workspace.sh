#!/usr/bin/env bash

rm -rf mace/codegen/models
rm -rf mace/codegen/engine
rm -rf mace/codegen/opencl

for d in build/*; do
    if [[ "$d" != "build/cmake-build" ]]; then
	echo "remove $d"
        rm -rf "$d"
    fi
done

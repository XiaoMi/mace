#! /bin/bash

cd docker/mace-micro-dev

docker build . -f mace-micro-dev.dockerfile --tag mace-micro-dev

cd ../..
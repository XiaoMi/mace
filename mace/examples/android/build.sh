#!/usr/bin/env bash

set -e -u -o pipefail

pushd ../../../

python tools/converter.py convert --config=mace/examples/android/mobilenet.yml
cp -rf builds/mobilenet/include mace/examples/android/macelibrary/src/main/cpp/
cp -rf builds/mobilenet/model mace/examples/android/macelibrary/src/main/cpp/

bash tools/build-standalone-lib.sh
cp -rf builds/include mace/examples/android/macelibrary/src/main/cpp/
cp -rf builds/lib mace/examples/android/macelibrary/src/main/cpp/

popd

if [ $# -eq 1 ] && [ $1 == "build" ]; then
    ./gradlew build
else
    ./gradlew installAppRelease
fi

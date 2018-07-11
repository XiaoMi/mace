#!/usr/bin/env bash

set -e -u -o pipefail

pushd ../../../
python tools/converter.py build --config=docs/user_guide/models/demo_app_models.yml

cp -r builds/mobilenet/include mace/examples/android/macelibrary/src/main/cpp/
cp -r builds/mobilenet/lib mace/examples/android/macelibrary/src/main/cpp/

popd

./gradlew installAppRelease

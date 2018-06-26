Android Demo
------------
------------

Try it by downloading the [APK](https://cnbj1.fds.api.xiaomi.com/mace/demo/mace_android_demo.apk).

How to build
---------------
* Build native library, please refer to [docs](docs) for more information.
    ```
    # Execute following commands from the project's root directory
    python tools/converter.py build --config=docs/getting_started/models/demo_app_models.yaml
    cp -r build/mobilenet/include/ mace/examples/android/macelibrary/src/main/cpp/
    cp -r build/mobilenet/lib/ mace/examples/android/macelibrary/src/main/cpp/
    ```
* Build APK
  * Import the `mace/examples/android` directory as a new Android Studio project and select `install run`
  * Or build it with gradle:
    ```
    cd mace/exampls/android
    ./gradlew installAppRelease
    ```
* You can also build native library and APK with `mace/examples/android/build.sh`
    ```
    cd mace/exampls/android
    ./build.sh
    ```

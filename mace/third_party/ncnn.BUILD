genrule(
    name = "cmake",
    outs = [
        "build-android-aarch64/install/lib/libncnn.a",
        "build-android-aarch64/install/include/blob.h",
        "build-android-aarch64/install/include/cpu.h",
        "build-android-aarch64/install/include/layer.h",
        "build-android-aarch64/install/include/mat.h",
        "build-android-aarch64/install/include/net.h",
        "build-android-aarch64/install/include/opencv.h",
        "build-android-aarch64/install/include/layer_type_enum.h",
        "build-android-aarch64/install/include/platform.h",
    ],
    cmd = "pwd; echo $$(dirname $(location cpu.h)); mkdir -p build-android-aarch64; pushd build-android-aarch64; cmake -DCMAKE_TOOLCHAIN_FILE=../external/ncnn/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_FORCE_ARM_BUILD=OFF -DANDROID_STL_FORCE_FEATURES=OFF ../external/ncnn && make && make install; popd",
)

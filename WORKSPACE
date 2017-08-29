workspace(name = "mace")

# proto_library rules implicitly depend on @com_google_protobuf//:protoc,
# which is the proto-compiler.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    urls = ["http://v9.git.n.xiaomi.com/deep-learning/protobuf/repository/archive.zip?ref=c7457ef65a7a8584b1e3bd396c401ccf8e275ffa"],
    strip_prefix = "protobuf-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa",
    sha256 = "0a54cae83b77f4b54b7db4eaebadd81fbe91655e84a1ef3f6d29116d75f3a45f",
)

# cc_proto_library rules implicitly depend on @com_google_protobuf_cc//:cc_toolchain,
# which is the C++ proto runtime (base classes and common utilities).
http_archive(
    name = "com_google_protobuf_cc",
    urls = ["http://v9.git.n.xiaomi.com/deep-learning/protobuf/repository/archive.zip?ref=c7457ef65a7a8584b1e3bd396c401ccf8e275ffa"],
    strip_prefix = "protobuf-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa",
    sha256 = "0a54cae83b77f4b54b7db4eaebadd81fbe91655e84a1ef3f6d29116d75f3a45f",
)

new_http_archive(
    name = "gtest",
    url = "http://v9.git.n.xiaomi.com/deep-learning/googletest/repository/archive.zip?ref=release-1.8.0",
    strip_prefix = "googletest-release-1.8.0-ec44c6c1675c25b9827aacd08c02433cccde7780",
    sha256 = "a0b43a0a43cda0cc401a46d75519d961ef27f6674d4126366e47d9c946c4bbcd",
    build_file = "mace/third_party/gtest.BUILD",
)

# Set up Android NDK
android_ndk_repository(
    name = "androidndk",
    # Android 5.0
    api_level = 21
)

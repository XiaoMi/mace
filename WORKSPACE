workspace(name = "mace")

# proto_library rules implicitly depend on @com_google_protobuf//:protoc,
# which is the proto-compiler.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "40d39d97a7b514b3e34daef732f822eca0081960b269863f5b573db5548cb237",
    strip_prefix = "protobuf-3.4.0rc3",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/protobuf/protobuf-3.4.0rc3.zip",
        "https://github.com/google/protobuf/archive/v3.4.0rc3.zip"
    ],
)

new_http_archive(
    name = "gtest",
    build_file = "mace/third_party/googletest/googletest.BUILD",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    strip_prefix = "googletest-release-1.8.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/googletest/googletest-release-1.8.0.zip",
        "https://github.com/google/googletest/archive/release-1.8.0.zip"
    ],
)

new_http_archive(
    name = "opencl_headers",
    build_file = "mace/third_party/opencl-headers/opencl-headers.BUILD",
    sha256 = "5dc7087680853b5c825360fc04ca26534f4b9f22ac114c4d3a306bfbec3cd0f2",
    strip_prefix = "OpenCL-Headers-master",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-Headers/OpenCL-Headers-master.zip",
        "https://github.com/KhronosGroup/OpenCL-Headers/archive/master.zip"
    ],
)

new_http_archive(
    name = "opencl_clhpp",
    build_file = "mace/third_party/opencl-clhpp/opencl-clhpp.BUILD",
    sha256 = "d4eb63372ad31f7efcae626852f75f7929ff28d1cabb5f50ef11035963a69b46",
    strip_prefix = "OpenCL-CLHPP-2.0.10",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-CLHPP/OpenCL-CLHPP-2.0.10.zip",
        "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v2.0.10.zip"
    ],
)

new_http_archive(
    name = "half",
    build_file = "mace/third_party/half/half.BUILD",
    sha256 = "cdd70d3bf3fe091b688e7ab3f48471c881a197d2c186c95cca8bf156961fb41c",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/half/half-1.12.0.zip",
        "https://jaist.dl.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip"
    ],
)

new_http_archive(
    name = "six_archive",
    build_file = "mace/third_party/six/six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/six/six-1.10.0.tar.gz",
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)

git_repository(
    name = "com_github_gflags_gflags",
    #tag    = "v2.2.0",
    commit = "30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e",  # v2.2.0 + fix of include path
    remote = "https://github.com/gflags/gflags.git",
)

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

bind(
    name = "gflags_nothreads",
    actual = "@com_github_gflags_gflags//:gflags_nothreads",
)

# Set up Android NDK
android_ndk_repository(
    name = "androidndk",
    # Android 5.0
    api_level = 21,
)

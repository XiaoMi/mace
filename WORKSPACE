workspace(name = "mace")

# proto_library rules implicitly depend on @com_google_protobuf//:protoc,
# which is the proto-compiler.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "542703acadc3f690d998f4641e1b988f15ba57ebca05fdfb1cd9095bec007948",
    strip_prefix = "protobuf-3.4.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/protobuf/protobuf-3.4.0.zip",
        "https://github.com/google/protobuf/archive/v3.4.0.zip"
    ],
)

new_http_archive(
    name = "gtest",
    build_file = "third_party/googletest/googletest.BUILD",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    strip_prefix = "googletest-release-1.8.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/googletest/googletest-release-1.8.0.zip",
        "https://github.com/google/googletest/archive/release-1.8.0.zip"
    ],
)

new_http_archive(
    name = "opencl_headers",
    build_file = "third_party/opencl-headers/opencl-headers.BUILD",
    sha256 = "5dc7087680853b5c825360fc04ca26534f4b9f22ac114c4d3a306bfbec3cd0f2",
    strip_prefix = "OpenCL-Headers-master",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-Headers/OpenCL-Headers-master.zip",
        "https://github.com/KhronosGroup/OpenCL-Headers/archive/master.zip"
    ],
)

new_http_archive(
    name = "opencl_clhpp",
    build_file = "third_party/opencl-clhpp/opencl-clhpp.BUILD",
    sha256 = "dab6f1834ec6e3843438cc0f97d63817902aadd04566418c1fcc7fb78987d4e7",
    strip_prefix = "OpenCL-CLHPP-4c6f7d56271727e37fb19a9b47649dd175df2b12",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-CLHPP/OpenCL-CLHPP-4c6f7d56271727e37fb19a9b47649dd175df2b12.zip",
        "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/4c6f7d56271727e37fb19a9b47649dd175df2b12.zip"
    ],
)

new_http_archive(
    name = "half",
    build_file = "third_party/half/half.BUILD",
    sha256 = "0f514a1e877932b21dc5edc26a148ddc700b6af2facfed4c030ca72f74d0219e",
    strip_prefix = "half-code-356-trunk",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/half/half-code-356-trunk.zip",
        "https://sourceforge.net/code-snapshots/svn/h/ha/half/code/half-code-356-trunk.zip"
    ],
)

new_http_archive(
    name = "six_archive",
    build_file = "third_party/six/six.BUILD",
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

http_archive(
    # v2.2.0 + fix of include path
    name = "com_github_gflags_gflags",
    sha256 = "16903f6bb63c00689eee3bf7fb4b8f242934f6c839ce3afc5690f71b712187f9",
    strip_prefix = "gflags-30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gflags/gflags-30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e.zip",
        "https://github.com/gflags/gflags/archive/30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e.zip"
    ],
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

workspace(name = "mace")

# proto_library rules implicitly depend on @com_google_protobuf//:protoc,
# which is the proto-compiler.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    urls = ["http://v9.git.n.xiaomi.com/deep-computing/protobuf/repository/archive.zip?ref=c7457ef65a7a8584b1e3bd396c401ccf8e275ffa"],
    strip_prefix = "protobuf-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa",
    sha256 = "0a54cae83b77f4b54b7db4eaebadd81fbe91655e84a1ef3f6d29116d75f3a45f",
)

# cc_proto_library rules implicitly depend on @com_google_protobuf_cc//:cc_toolchain,
# which is the C++ proto runtime (base classes and common utilities).
http_archive(
    name = "com_google_protobuf_cc",
    urls = ["http://v9.git.n.xiaomi.com/deep-computing/protobuf/repository/archive.zip?ref=c7457ef65a7a8584b1e3bd396c401ccf8e275ffa"],
    strip_prefix = "protobuf-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa-c7457ef65a7a8584b1e3bd396c401ccf8e275ffa",
    sha256 = "0a54cae83b77f4b54b7db4eaebadd81fbe91655e84a1ef3f6d29116d75f3a45f",
)

new_http_archive(
    name = "gtest",
    url = "http://v9.git.n.xiaomi.com/deep-computing/googletest/repository/archive.zip?ref=release-1.8.0",
    strip_prefix = "googletest-release-1.8.0-ec44c6c1675c25b9827aacd08c02433cccde7780",
    sha256 = "a0b43a0a43cda0cc401a46d75519d961ef27f6674d4126366e47d9c946c4bbcd",
    build_file = "mace/third_party/gtest.BUILD",
)

new_http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "mace/third_party/six.BUILD",
)
bind(
      name = "six",
      actual = "@six_archive//:six",
)

new_http_archive(
    name = "opencl_headers",
    urls = ["http://v9.git.n.xiaomi.com/deep-computing/OpenCL-Headers/repository/archive.zip?ref=master"],
    strip_prefix = "OpenCL-Headers-master-f039db6764d52388658ef15c30b2237bbda49803",
    sha256 = "439dbdb4e7a02a218dd90d82170c9f7671487cd0e626a20e136690a91f173ad2",
    build_file = "mace/third_party/opencl-headers.BUILD",
)

# Set up Android NDK
android_ndk_repository(
    name = "androidndk",
    # Android 5.0
    api_level = 21
)

git_repository(
    name   = "com_github_gflags_gflags",
    #tag    = "v2.2.0",
    commit = "30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e", # v2.2.0 + fix of include path
    remote = "https://github.com/gflags/gflags.git"
)

bind(
    name   = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

bind(
    name   = "gflags_nothreads",
    actual = "@com_github_gflags_gflags//:gflags_nothreads",
)

workspace(name = "mace")

http_archive(
    name = "org_tensorflow",
    urls = ["http://v9.git.n.xiaomi.com/deep-learning/tensorflow/repository/archive.zip?ref=v1.3.0"],
    strip_prefix = "tensorflow-v1.3.0-9e76bf324f6bac63137a02bb6e6ec9120703ea9b",
    sha256 = "97049d3a59a77858e12c55422bd129261b14e869a91aebcdcc39439393c00dc7",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "60fc6977908f999b23ca65698c2bb70213403824a84f7904310b6000d78be9ce",
    strip_prefix = "rules_closure-5ca1dab6df9ad02050f7ba4e816407f88690cf7d",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",  # 2017-02-03
        "https://github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",
    ],
)

# Import all of the tensorflow dependencies.
load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace(tf_repo_name = "org_tensorflow")

new_http_archive(
    name = "ncnn",
    urls = ["http://v9.git.n.xiaomi.com/deep-learning/ncnn/repository/archive.zip?ref=bazel-fix"],
    strip_prefix = "ncnn-bazel-fix-ce5e416164545e1ab37fe3544502624f605ca234/src",
    sha256 = "e6d76356179bcdbb988279f0b42ab050c8af55970e1ad767787ad21d5b7aad51",
    build_file = "mace/third_party/ncnn.BUILD",
)

# Set up Android NDK
android_ndk_repository(
    name = "androidndk",
    # Android 4.0
    api_level = 14
)

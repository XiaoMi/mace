"""Repository rule for opencl encrypt kernel autoconfiguration, borrow from tensorflow
"""

def _opencl_encrypt_kernel_impl(repository_ctx):
    repository_ctx.template(
        "BUILD.bazel",
        Label("//repository/opencl-kernel:BUILD.bazel.tpl"),
    )

    mace_root_path = str(repository_ctx.path(Label("@mace//:BUILD.bazel")))[:-len("BUILD.bazel")]
    generated_files_path = repository_ctx.path("gen")

    ret = repository_ctx.execute(
        ["test", "-f", "%s/.git/logs/HEAD" % mace_root_path],
    )
    if ret.return_code == 0:
        unused_var = repository_ctx.path(Label("//:.git/HEAD"))
    ret = repository_ctx.execute(
        ["test", "-f", "%s/.git/refs/heads/master" % mace_root_path],
    )
    if ret.return_code == 0:
        unused_var = repository_ctx.path(Label("//:.git/refs/heads/master"))

    ret = repository_ctx.execute(
        ["test", "-f", "%s/mace/ops/opencl/cl/common.h" % mace_root_path],
    )
    if ret.return_code == 0:
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/activation.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/addn.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/batch_norm.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/batch_to_space.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/bias_add.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/buffer_to_image.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/buffer_transform.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/channel_shuffle.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/common.h"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/concat.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/conv_2d.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/conv_2d_1x1.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/conv_2d_1x1_buffer.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/conv_2d_3x3.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/conv_2d_buffer.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/crop.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/deconv_2d.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/depthwise_deconv2d.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/depth_to_space.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/depthwise_conv2d.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/depthwise_conv2d_buffer.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/eltwise.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/fully_connected.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/lstmcell.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/matmul.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/pad.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/pooling.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/pooling_buffer.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/reduce.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/resize_bicubic.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/resize_bilinear.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/resize_nearest_neighbor.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/split.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/softmax.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/softmax_buffer.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/space_to_batch.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/space_to_depth.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/sqrdiff_mean.cl"))
        unused_var = repository_ctx.path(Label("//:mace/ops/opencl/cl/winograd_transform.cl"))

    python_bin_path = repository_ctx.which("python")

    repository_ctx.execute([
        python_bin_path,
        "%s/mace/python/tools/encrypt_opencl_codegen.py" % mace_root_path,
        "--cl_kernel_dir=%s/mace/ops/opencl/cl" % mace_root_path,
        "--output_path=%s/encrypt_opencl_kernel" % generated_files_path,
    ], quiet = False)

encrypt_opencl_kernel_repository = repository_rule(
    implementation = _opencl_encrypt_kernel_impl,
)

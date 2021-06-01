# -*- Python -*-

def if_linux_base(a, default_value = []):
    return select({
        "//mace:linux_base": a,
        "//conditions:default": default_value,
    })

def if_android(a, default_value = []):
    return select({
        "//mace:android": a,
        "//conditions:default": default_value,
    })

def if_linux(a, default_value = []):
    return select({
        "//mace:linux": a,
        "//conditions:default": default_value,
    })

def if_darwin(a, default_value = []):
    return select({
        "//mace:darwin": a,
        "//conditions:default": default_value,
    })

def if_android_armv7(a):
    return select({
        "//mace:android_armv7": a,
        "//conditions:default": [],
    })

def if_android_arm64(a):
    return select({
        "//mace:android_arm64": a,
        "//conditions:default": [],
    })

def if_arm_linux_aarch64(a):
    return select({
        "//mace:arm_linux_aarch64": a,
        "//conditions:default": [],
    })

def if_arm_linux_armhf(a):
    return select({
        "//mace:arm_linux_armhf": a,
        "//conditions:default": [],
    })

def if_cpu_enabled(a, default_value = []):
    return select({
        "//mace:cpu_enabled": a,
        "//conditions:default": default_value,
    })

def if_neon_enabled(a, default_value = []):
    return select({
        "//mace:neon_enabled": a,
        "//conditions:default": default_value,
    })

def if_hexagon_enabled(a, default_value = []):
    return select({
        "//mace:hexagon_enabled": a,
        "//conditions:default": default_value,
    })

def if_not_hexagon_enabled(a, default_value = []):
    return select({
        "//mace:hexagon_enabled": default_value,
        "//conditions:default": a,
    })

def if_hta_enabled(a, default_value = []):
    return select({
        "//mace:hta_enabled": a,
        "//conditions:default": default_value,
    })

def if_hexagon_or_hta_enabled(a, default_value = []):
    return select({
        "//mace:hexagon_enabled": a,
        "//mace:hta_enabled": a,
        "//conditions:default": default_value,
    })

def if_apu_enabled(a, default_value = []):
    return select({
        "//mace:apu_enabled": a,
        "//conditions:default": default_value,
    })

def if_not_apu_enabled(a, default_value = []):
    return select({
        "//mace:apu_enabled": default_value,
        "//conditions:default": a,
    })

def apu_version_select(v1, v2, v3, v4):
    return select({
        "//mace:apu_v1": v1,
        "//mace:apu_v2": v2,
        "//mace:apu_v3": v3,
        "//mace:apu_v4": v4,
        "//conditions:default": [],
    })

def if_opencl_enabled(a, default_value = []):
    return select({
        "//mace:opencl_enabled": a,
        "//conditions:default": default_value,
    })

def if_quantize_enabled(a):
    return select({
        "//mace:quantize_enabled": a,
        "//conditions:default": [],
    })

def if_bfloat16_enabled(a):
    return select({
        "//mace:bfloat16_enabled": a,
        "//conditions:default": [],
    })

def if_fp16_enabled(a):
    return select({
        "//mace:fp16_enabled": a,
        "//conditions:default": [],
    })

def if_rpcmem_enabled(a, default_value = []):
    return select({
        "//mace:rpcmem_enabled": a,
        "//conditions:default": default_value,
    })

def if_opencl_and_rpcmem_enabled(a, default_value = []):
    return select({
        "//mace:opencl_and_rpcmem_enabled": a,
        "//conditions:default": default_value,
    })

def mace_version_genrule():
    native.genrule(
        name = "mace_version_gen",
        srcs = [str(Label("@local_version_config//:gen/version"))],
        outs = ["version/version.cc"],
        cmd = "cat $(SRCS) > $@;",
    )

def encrypt_opencl_kernel_genrule():
    srcs = [
        str(Label(
            "@local_opencl_kernel_encrypt//:gen/encrypt_opencl_kernel.cc",
        )),
        str(Label(
            "@local_opencl_kernel_encrypt//:gen/encrypt_opencl_kernel.h",
        )),
    ]
    outs = ["opencl/encrypt_opencl_kernel.cc", "opencl/encrypt_opencl_kernel.h"]
    native.genrule(
        name = "encrypt_opencl_kernel_gen",
        srcs = srcs,
        outs = outs,
        cmd = " && ".join([
            "cat $(location %s) > $(location %s)" % (srcs[i], outs[i])
            for i in range(0, len(outs))
        ]),
    )

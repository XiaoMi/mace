package(default_visibility = ['//visibility:public'])

cc_library(
    name = "headers_tools_target",
    hdrs = glob([
        "target/hexagon/include/**/*.h",
    ]),
    strip_include_prefix = "target/hexagon/include/",
)

filegroup(
    name = 'gcc',
    srcs = [
        'bin/hexagon-clang',
    ],
)

filegroup(
    name = 'ar',
    srcs = [
        'bin/hexagon-ar',
    ],
)

filegroup(
    name = 'ld',
    srcs = [
        'bin/hexagon-link',
    ],
)

filegroup(
    name = 'nm',
    srcs = [
        'bin/hexagon-nm',
    ],
)

filegroup(
    name = 'objcopy',
    srcs = [
        'bin/hexagon-elfcopy',
    ],
)

filegroup(
    name = 'objdump',
    srcs = [
        'bin/hexagon-llvm-objdump',
    ],
)

filegroup(
    name = 'strip',
    srcs = [
        'bin/hexagon-strip',
    ],
)

filegroup(
    name = 'as',
    srcs = [
        'bin/hexagon-llvm-mc',
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "libexec/**",
        "lib/**",
        "include/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)

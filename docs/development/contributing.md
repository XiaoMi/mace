Contributing guide
==================

License
-------

The source file should contain a license header. See the existing files
as the example.

Python coding style
-------------------

Changes to Python code should conform to [PEP8 Style Guide for Python
Code](https://www.python.org/dev/peps/pep-0008/).

You can use [pycodestyle](ihttps://github.com/PyCQA/pycodestyle) to check the
style.

C++ coding style
----------------

Changes to C++ code should conform to [Google C++ Style
Guide](https://google.github.io/styleguide/cppguide.html).

You can use cpplint to check the style and use clang-format to format
the code:

```sh
clang-format -style="{BasedOnStyle: google,            \
                      DerivePointerAlignment: false,   \
                      PointerAlignment: Right,         \
                      BinPackParameters: false}" $file
```

C++ logging guideline
---------------------

VLOG is used for verbose logging, which is configured by environment variable
`MACE_CPP_MIN_VLOG_LEVEL`. The guideline of VLOG level is as follows:

```
0. Ad hoc debug logging, should only be added in test or temporary ad hoc
   debugging
1. Important network level Debug/Latency trace log (Op run should never
   generate level 1 vlog)
2. Important op level Latency trace log
3. Unimportant Debug/Latency trace log
4. Verbose Debug/Latency trace log
```

C++ marco
----------
C++ macros should start with `MACE_`, except for most common ones like `LOG`
and `VLOG`.

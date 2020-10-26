#!/usr/bin/env sh

set -e

cpplint --linelength=80 --counting=detailed $(find mace -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=include $(find include -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccutils $(find test/ccutils -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccunit $(find test/ccunit -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccbenchmark $(find test/ccbenchmark -name "*.h" -or -name "*.cc")

cpplint --linelength=80 --counting=detailed --filter=-build/include_what_you_use $(find micro/base  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed $(find micro/framework  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed $(find micro/include  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed $(find micro/model  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --filter=-build/include_what_you_use $(find micro/ops  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed $(find micro/port  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --filter=-build/include_what_you_use $(find micro/test \( -path micro/test/ccbenchmark/codegen -or -path micro/test/ccbaseline/codegen \) -prune -o  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed $(find micro/tools  -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --filter=-build/include_subdir $(find micro/examples \( -path micro/examples/classifier/mbed-os -or -path micro/examples/classifier/data -or -path micro/examples/classifier/install -or -path micro/examples/classifier/BUILD \) -prune  -name "*.cc" -or -name "*.h")

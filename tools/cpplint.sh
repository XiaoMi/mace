#!/usr/bin/env sh

set -e

cpplint --linelength=80 --counting=detailed $(find mace -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=include $(find include -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccutils $(find test/ccutils -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccunit $(find test/ccunit -name "*.h" -or -name "*.cc")
cpplint --linelength=80 --counting=detailed --root=test/ccbenchmark $(find test/ccbenchmark -name "*.h" -or -name "*.cc")

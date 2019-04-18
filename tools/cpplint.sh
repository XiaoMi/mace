#!/usr/bin/env sh

set -e

curl -o cpplint.py https://raw.githubusercontent.com/google/styleguide/gh-pages/cpplint/cpplint.py
python cpplint.py --linelength=80 --counting=detailed $(find mace -name "*.h" -or -name "*.cc")
python cpplint.py --linelength=80 --counting=detailed --root=include $(find include -name "*.h" -or -name "*.cc")
python cpplint.py --linelength=80 --counting=detailed --root=test/ccutils $(find test/ccutils -name "*.h" -or -name "*.cc")
python cpplint.py --linelength=80 --counting=detailed --root=test/ccunit $(find test/ccunit -name "*.h" -or -name "*.cc")
python cpplint.py --linelength=80 --counting=detailed --root=test/ccbenchmark $(find test/ccbenchmark -name "*.h" -or -name "*.cc")
rm cpplint.py

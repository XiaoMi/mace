#!/bin/bash

clang-format \
  -style="{BasedOnStyle: google,    \
           DerivePointerAlignment: false, \
           PointerAlignment: Right, \
           BinPackParameters: false}"  -i $1

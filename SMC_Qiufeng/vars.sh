#!/bin/bash

# EDIT THESE VARIABLES IF LLVM BUILD DIR IS NOT IN HOME
WORKING_DIR="$PWD"
BENCHMARKS="$WORKING_DIR/rodinia_3.1/cuda"
BUILD_DIR="$HOME/llvm/build"
TOOLS_EXTRA="$HOME/llvm/tools/clang/tools/extra"

if [ ! -d "$BENCHMARKS" ]; then
    echo "Warning: Rodinia benchmarks directory \"$BENCHMARKS\" does not exist (only used in run.sh)";
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: LLVM build directory \"$BUILD_DIR\" does not exist";
    exit 1;
fi

if [ ! -d "$TOOLS_EXTRA" ]; then
    echo "Creating llvm/tools/extra directory";
    mkdir $TOOLS_EXTRA;
fi

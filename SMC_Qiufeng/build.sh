#!/bin/bash

source vars.sh

#When editing smc.cpp locally, need to copy over right before build
cp $WORKING_DIR/smc/smc.cpp $TOOLS_EXTRA/smc/

# Compile tools
cd $BUILD_DIR
ninja

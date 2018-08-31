#!/bin/bash

source vars.sh;

# Setup clang extra sub-folder
if [ ! -d "$TOOLS_EXTRA" ]; then
    mkdir $TOOLS_EXTRA
fi

if [ ! -d "$TOOLS_EXTRA/smc" ]; then
    mkdir $TOOLS_EXTRA/smc
fi

# cp $WORKING_DIR/smc.cpp $TOOLS_EXTRA/smc #Cmake needs to see the file here, but we'll still copy it for the build stage (so no re-con figure is required)
# cp $WORKING_DIR/CMakeLists.txt $TOOLS_EXTRA/smc
cp -r $WORKING_DIR/smc $TOOLS_EXTRA/

# Add tool details to common CMakeList file
if [ -f $TOOLS_EXTRA/CMakeLists.txt ]; then
    #If file exists, grep it for the command we're about to
    #insert. This ensures configure.sh can be called multiple times
    #without crashing.  
    if grep add_subdirectory\(smc\) $TOOLS_EXTRA/CMakeLists.txt; then
	echo "Already inserted smc subdirectory command in lang-tools-extras' CMakeLists.txt file"
    else
	echo 'add_subdirectory(smc)' >> "$TOOLS_EXTRA/CMakeLists.txt";
    fi
else
    #File does not exist, create
    echo 'add_subdirectory(smc)' >> "$TOOLS_EXTRA/CMakeLists.txt";
fi

cd $BUILD_DIR;
cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCMAKE_BUILD_TYPE=Release

#! /bin/bash

rootLlvm=~/llvm
flPath=free-launch #free-launch should be in the same directory

# scriptHelper() {
    # echo "
    # The script expectes two arguments.
    # 1. llvm_path: This is the path where the llvm root directory is present. The
    #               path includes llvm directory name as well. The script expects 
    #               the following directory structure:
    #                    <llvm_path>
    #                    |-- tools
    #                        |-- clang
    #                            |-- tools
    #                                |-- extra
    #                                    |-- <free-launch would come here>

    # 2. free_launch_path: This is the path where the free-launch directory is 
    #                      present. The path includes the free-launch directory as
    #                      well.

    # Example Invocation:
    # ./setup.sh -llvm_path ~/llvm -free_launch_path ~/Git/Compiler/free-launch
    # "
# }

# Reference for parsing arguments are taken from 
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash/13359121?noredirect=1#comment29656357_13359121
# while [ "$#" -gt 0 ]; do
#   case "$1" in
#     -llvm_path) 
#         rootLlvm="$2" 
#         shift 2;;
#     -free_launch_path) 
#         flPath="$2" 
#         shift 2;;
#     *) scriptHelper;;
#   esac
# done

# if [ "$rootLlvm" == "" ] || [ "$flPath" == "" ]; then
#     scriptHelper
#     exit
# fi

extras=$rootLlvm/tools/clang/tools/extra
cp -r $flPath $extras

#Check whether to insert CMakeLists.txt entry
if [ -f $extras/CMakeLists.txt ]; then
    if grep add_subdirectory\(free-launch\) $extras/CMakeLists.txt; then
	echo "Already inserted free-launch subdirectory command in lang-tools-extras' CMakeLists.txt file"
    else
	echo "add_subdirectory(free-launch)" >> $extras/CMakeLists.txt
    fi
else
    echo "add_subdirectory(free-launch)" >> $extras/CMakeLists.txt
fi

cd $rootLlvm/build
#cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCMAKE_BUILD_TYPE=Release
ninja

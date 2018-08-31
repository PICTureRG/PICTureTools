#! /bin/bash

bin=
input=
transform=
header=~/PICTureTools/FreeLaunch_jsingh8/headers
args=
fileName=
folderName=
transformName=

scriptHelper() {
    echo "
    The script takes 3 mandatory arguments and 1 options argument.
    Mandatory Arguments:
        1) binary: The full path identifying the binary tool file.
        2) input: The full path representing the file that needs to be transformed. The file should be a compilable file.
        3) transformation: The type of transformation to do. User can choose from T1, T2, T3 or T4.

    Optional Arguments:
        1) compiler_args: The arguments that needs to be passed to the clang compiler.

    Example:
    ./run.sh \
        -binary ~/free-launch-1 \
        -input ~/test_folder/main.cu \
        -transformation T1 \
        -header ~/headerFolder \
        -compiler_args \"-- --cuda-host-only -I ~/some_folder/include\"
    "
    # 4) header: The folder containing header files pertaining to the transformation.
    exit
}

# Reference for parsing arguments are taken from 
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash/13359121?noredirect=1#comment29656357_13359121
while [ "$#" -gt 0 ]; do
  case "$1" in
    -binary) 
        bin="$2" 
        shift 2;;
    -input) 
        input="$2" 
        shift 2;;
    -transformation) 
        transform="$2" 
        if [ "$transform" != "T1" ] && [ "$transform" != "T2" ] && [ "$transform" != "T3" ] && [ "$transform" != "T4" ]; then
            scriptHelper
        fi
        shift 2;;
    # -header) 
    #     header="$2" 
    #     shift 2;;
    -compiler_args) 
        args="$2" 
        shift 2;;
    *) scriptHelper;;
  esac
done

if [ "$bin" == "" ] || [ "$input" == "" ] || [ "$transform" == "" ] || [ "$header" == "" ]; then
    scriptHelper
fi

getFolderName() {
    temp=$1
    temp1=
    folderName=

    while true
    do
        temp1=`echo $temp | awk -F "/" '{ if (NF > 1) { print $1 "/"} else { print $1}}'`
        temp=${temp/$temp1/}
        if [ "$temp" == "" ]; then
            fileName=$temp1
            break
        fi
        folderName=`echo "$folderName$temp1"`
    done
}

getTransformFolder() {
    temp=$1
    temp1=
    temp2=
    transformName=
    transformType=$2
    
    while true
    do
        temp1=`echo $temp | awk -F "/" '{ if (NF > 1) { print $1 "/"} else { print $1}}'`
        temp2=`echo $temp | awk -v type="$transformType" -F "/" '{ if (NF > 2) { print $1 "/"} else { print $1"_"type}}'`
        temp=${temp/$temp1/}
        transformName=`echo "$transformName$temp2"`
        if [ "$temp" == "" ]; then
            break
        fi
    done
}

makeDir () {
    mkdir -p $1
}

copyFiles() {
    cp $1/* $2/
}   

# Arg 1: File Name
# Arg 2: Transformation
run() {
    getFolderName $1
    getTransformFolder $folderName $2
    makeDir $transformName
    copyFiles $folderName $transformName
}

# Arg1: Binary
# Arg2: File to compile
execute_compiler() {
    $1 $2 $args
}

transform() {
    run $input $transform
    execute_compiler $bin "${transformName}/${fileName}"

    if [ "$transform" == "T1" ]; then 
        cp ${header}/freeLaunch_T1.h ${transformName}/
    elif [ "$transform" == "T2" ]; then
        cp ${header}/freeLaunch_T2.h ${transformName}/
    elif [ "$transform" == "T3" ]; then
        cp ${header}/freeLaunch_T3.h ${transformName}/
    fi
}

transform

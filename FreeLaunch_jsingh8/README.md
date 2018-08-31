# Free-Launch #

TODO: Intro

Overview of directory structure. 

```
|-- free-launch.pdf           -> The project report
|-- run.sh                    -> script to be used for running the tool.
|-- setup.sh                  -> script for tool installation.
|-- free-launch               -> Clang tool source code.
    |-- free-launch-1
        |-- T1.cpp
        |-- CMakeLists.txt
    |-- free-launch-2
        |-- T2.cpp
        |-- CMakeLists.txt
    |-- free-launch-3
        |-- T3.cpp
        |-- CMakeLists.txt
    |-- free-launch-4
        |-- T4.cpp
        |-- CMakeLists.txt
    |-- util.h
    |-- CMakeLists.txt
|-- test                      -> folder with 12 test case folders.
```

## Compilation ##

#### Prerequisites ####

This tool requires a LLVM-CLANG source build, as well as the clang-reorder-fields tool from clang-tools-extra. Follow these steps to setup llvm in the home directory. 
```
#LLVM
cd
git clone https://git.llvm.org/git/llvm.git/

#CLANG
cd llvm/tools
git clone https://git.llvm.org/git/clang.git/

#clang-reorder-fields
cd clang/tools/
mkdir extra
cd extra
git clone https://github.com/llvm-mirror/clang-tools-extra
mv clang-tools-extra/clang-reorder-fields ./
rm -rf clang-tools-extra # Remove unnecessary extras
echo "add_subdirectory(clang-reorder-fields)" >> CMakeLists.txt # <<< only run once!

#Build LLVM using ninja in Release mode (builds faster, still takes a while)
cd ../../../../
mkdir build
cd build
#Assumes you have ninja installed at /usr/bin/ninja
cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

__NOTE:__ This will build the LLVM-CLANG binaries, but there is
actually a problem in the CLANG code that has not been solved
yet. Dynamic parallelism in CUDA is a new development only supported
by architectures with compute capability 3.5 or higher. CLANG isn't
sure how to handle these types of kernels (for CUDA programmers, this
is a \_\_global\_\_ function calling another \_\_global\_\_ function). They
default to reporting an error when this is seen. The error is reported
by the following code in llvm/tools/clang/lib/Sema/SemaCUDA.cpp file
(around line 176 at the time of writing this 5/18/18).
```
if (CalleeTarget == CFT_Global &&
   (CallerTarget == CFT_Global || CallerTarget == CFT_Device))
  return CFP_Never;
```
This causes a crash that looks like:
```
~/PICTureTools/FreeLaunch_jsingh8/test/sanity/test_T1/main.cu:11:3: error: reference to
      __global__ function 'child' in __global__ function
  child<<<1, kDataLen>>>(z);
  ^
~/PICTureTools/FreeLaunch_jsingh8/test/sanity/test_T1/main.cu:5:17: note: 'child'
      declared here
__global__ void child(float *z) {
                ^
1 error generated when compiling for host.
Error while processing ~/PICTureTools/FreeLaunch_jsingh8/test/sanity/test_T1/main.cu.
```
The only work around for this at present is to comment out the code
that reports this error:
```
/*
if (CalleeTarget == CFT_Global &&
   (CallerTarget == CFT_Global || CallerTarget == CFT_Device))
  return CFP_Never;
*/
```

Lastly, rerun ninja to rebuild targets that depend on this file. Setup for Free-launch installation and usage should now be complete. 

## Free-launch Installation ##
The tool can be installed using the script setup.sh that is provided. __First__, if your LLVM is not installed at ~/llvm, change this default value in setup.sh! After doing so, it can be run as:
```
./setup.sh
#chmod u+x setup.sh # < run this if permission denied error, then try again
```

Once complete, the binary can be found in ~/llvm/build/bin/ assuming you followed the above llvm installation instructions. There would be 4 binaries generated. They are named free-launch-X where X is 1,2,3 and 4. These binaries correspond to each Transformer. 


## Tool Execution ##


The tool is expected to be used with the run.sh script that is provided.
The script takes 4 arguments out of which 3 are mandatory.
Mandatory Arguments:
```
1) binary: The full path identifying the binary tool file.
2) input: The full path representing the file that needs to be transformed. The file should be a compilable file.
3) transformation: The type of transformation to do. User can choose from T1, T2, T3 or T4.

Optional Arguments:
1) compiler_args: The arguments that needs to be passed to the clang 
   compiler.
```

For example, the tool can be utilzed using the command.
```
./run.sh \
    -binary ~/llvm/build/bin/free-launch-1 \
    -input ~/test_folder/main.cu \
    -transformation T1 \
    -header ~/headerFolder \
    -compiler_args "-- --cuda-host-only -I ~/some_folder/include"
```

If you encounter crazy errors, be sure you're running with:
```
-compiler_args "-- --cuda-host-only"
```
If you still get an error that says \_\_global\_\_ can't call another \_\_global\_\_, use:
```
-compiler_args "-- --cuda-host-only --cuda-gpu-arch=sm_50"
```


## Tool Output ##

The tool is expected to be run using run.sh script as described above.
On running the script, a successful exection would mean that a new folder
would be created with the transformed source code. For example, if the
source code to be transformed is present at A/B/C/main.c and the script is 
executed for T1 transformation, then a new folder C_T1 would be created
inside the folder B and the transformed main.c will be present in that 
folder. That is, the transformed file can be found at A/B/C_T1/main.c.

## Executing Test Cases ##

The code in the test section can be fed as input to the run.sh
script to generate the transformed code.

The correctness of the code can be tested by verifying if the 
transformations are as specified in the research paper "Free Launch: 
Optimizing GPU Dynamic Kernel Launches through Thread Reuse", provided
in this repository as free_launch.pdf. 

The correctness can then be tested by compiling the generated code and then
executing them. The code can be compiled with nvcc. The source code will
require the flags -arch sm_50 and -rdc true because of the subkernel 
launches. To have consistency, we advise using these flags for comparing
the outputs generated by the original code and transformed code.

## Known Bugs ##

There is 1 known bug in the code. T1, T2 and T3 transformations are not 
supported for cases when parent kernel is declared and then used. In
such cases, the tool does not transform the parent kernel declaration as
per the new arguments added to cache the child kernel parameters.

Several test cases also do not produce the same output as the original
program. (This would require some debugging from the original
author). 

## Platform Dependancy ##

The tool is developed to transform code that has subkernel launches. The 
tool does not have any platform dependancy. To execute code that has 
subkernel launches, it would be required to use CUDA with capabity >= 35.

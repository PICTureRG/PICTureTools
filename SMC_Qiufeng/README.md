# SMC #

__SM-Centric Transformation__

This project is designed to enable more flexible control over task assignment on the GPU. SMC is a source-to-source LLVM-CLANG addon that rewrites .cu files using the technique described in _ProjectReport.pdf_. 

#### Build structure ####

There are 2 building scripts within this directory, _configure.sh_ and _build.sh_. Since this project is an _addon_ to LLVM-CLANG, you will need LLVM-clang installed. The instructions to do this are found below.

The configure.sh script copies the smc.cpp and CMakeLists.txt files over to your llvm location (provided in vars.sh). It also inserts a line of code (or creates the file) into the llvm/tools/clang/extra/CMakeLists.txt file, allowing smc to be grabbed during normal cmake compilation of LLVM. The configure.sh script also launches the LLVM cmake configuration command (using Release mode, for better performance).

The build.sh script copies the local smc.cpp over to its corresponding llvm location, then launches the LLVM build. If LLVM has not been built, then several thousand targets will be compiled, which can take a while. Otherwise, only the SMC tool will be compiled. In this way, you can edit the smc.cpp file locally, then call build.sh to compile everything for quick testing. 

Note: Tested with CUDA 8.0.

## Compilation ##

__Edit the vars.sh file according to your llvm installation directory.__

__Note:__ we will use __$PROJECT__ to refer to the __PICTureTools/SMC_Qiufeng__ directory. 

#### Download and Configure LLVM

The following code downloads LLVM and clang to home directory, then runs a configure stage. This copies SMC files to the clang tools extra directory, inserting necessary CMakeLists.txt entries. Then it runs the LLVM configure. The LLVM configure must be run to discover the presence of the new SMC tool. 

```
cd
git clone https://git.llvm.org/git/llvm.git/
cd llvm/tools
git clone https://git.llvm.org/git/clang.git/
cd ..
cd $PROJECT/SMC_Qiufeng/
./configure.sh
```

#### Build project with LLVM

This step builds both LLVM and SMC using ninja. If LLVM is already built, it should only build the SMC file.

```
cd $PROJECT
./build.sh
```


## Running ##

The smc binary will be located in ~/llvm/build/bin (assuming llvm was built in ~). This directory can be added to your PATH variable:
```
export PATH=$PATH:~/llvm/build/bin
```

The _example_ directory (primarily the Makefile) provides an example of how to run the smc source-to-source translator.
The basic format of execution is:
```
smc my_file.cu -- --cuda-host-only -I my/include/dirs
```

The new source file will need to include the smc.h header file, which is included under the _smc_ directory here. That is, use:
```
nvcc my_file.cu -o my_exe -I path/to/smc -I my/other/includes
```

## Rodinia 3.1 Testing Setup ##

This project was tested with the Rodinia CUDA benchmarks. If you wish to run Rodinia tests, read on. 

#### rodinia_3.1/

Rodinia is too large to include here. It can be obtained using:
```
wget http://www.cs.virginia.edu/~kw5na/lava/Rodinia/Packages/Current/rodinia_3.1.tar.bz2
tar xvf rodinia_3.1.tar.bz2
```

We only needed the CUDA benchmarks, so we can delete the openMP and openCL directories and the portions of the Makefile that compile these targets (there are more bugs present in openMP and openCL, so removing their targets from the Makefile is necessary). 

__Required Setup__

- Ensure Rodinia project path has no spaces, this will break the Makefile. 

- Edit common/make.config: SDK_DIR is set as
```
    SDK_DIR = /usr/local/cuda-5.5/samples/
```
Depending on your cuda toolkit installation, this path may need to be changed. 

Errors that we fixed during our Rodinia build:

- Change [ch] to [c] in cuda/kmeans/Makefile and cuda/leukocyte/CUDA/Makefile 

- Change cuda/lavaMD/Makefile, cuda/lud/cuda/Makefile, and cuda/particlefilter/Makefile "sm_13" to "sm_20". 
  This is because sm_13 is not supported by later CUDA versions. 

- Added to top of _cuda/mummergpu/src/suffix-tree.cpp_ and _openmp/mummergpu/src/suffix-tree.cpp_ the line:
```
  #include <unistd.h>
```

After being able to compile the default cuda benchmark code, use the example design to construct smc versions of the benchmarks you would like to test.


## Bugs ##

There are several known issues with the source-to-source translation for this project. Certain special cases cannot be handled correctly by smc. These are listed in _ProjectReport.pdf_. 

Note that if there appear to be many compilation issues during the smc translation, that is likely because LLVM can't see certain definitions of variables or certain includes due to how the code is structured. For example, if the file being analyzed is never compiled, but is included in another file that has other includes, LLVM won't be able to see these includes. It will then send out a bunch of errors that can be safely ignored.

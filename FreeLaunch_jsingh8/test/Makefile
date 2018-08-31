# This Makefile builds and runs the sanity test.
# Set the BIN variable to the location of your free-launch-* binaries. 

# USAGE
# "make" - builds sanity test and all free launch transformations of it
# "make test_cuda" - runs normal cuda program
# "make test_freelaunch" - runs all transformed versions of the program
# "make clean" - removes build files


all: sanity/test/main

BIN=~/llvm/build/bin

sanity/test/main: ./sanity/test/main.cu
	nvcc $< -o ./sanity/test/main -rdc=true -arch=sm_35
	../run.sh -binary $(BIN)/free-launch-1 -transformation T1 -input $< -compiler_args "-- --cuda-host-only"
	nvcc ./sanity/test_T1/main.cu -o ./sanity/test_T1/main -rdc=true -arch=sm_35
	../run.sh -binary $(BIN)/free-launch-2 -transformation T2 -input $< -compiler_args "-- --cuda-host-only"
	nvcc ./sanity/test_T2/main.cu -o ./sanity/test_T2/main -rdc=true -arch=sm_35
	../run.sh -binary $(BIN)/free-launch-3 -transformation T3 -input $< -compiler_args "-- --cuda-host-only"
	nvcc ./sanity/test_T3/main.cu -o ./sanity/test_T3/main -rdc=true -arch=sm_35
	../run.sh -binary $(BIN)/free-launch-4 -transformation T4 -input $< -compiler_args "-- --cuda-host-only"
	nvcc ./sanity/test_T4/main.cu -o ./sanity/test_T4/main -rdc=true -arch=sm_35

test_cuda: sanity/test/main
	./sanity/test/main

test_freelaunch: sanity/test_T1/main sanity/test_T2/main sanity/test_T3/main sanity/test_T4/main
	./sanity/test_T1/main
	./sanity/test_T2/main
	./sanity/test_T3/main
	./sanity/test_T4/main


clean:
	rm -r ./sanity/test_T* ./sanity/test/main

# Type 'make' to produce 2 executables, matrix_mul and
# matrix_mul_smc. This file is to demonstrate the simplest
# functionality of smc. The output of matrix_mul_smc has not been
# checked for correctness, and appears to report far too high
# GFLOPS. Suspecting a bug. 

all: default smc

#Normal compilatin
default:
	nvcc matrix_mul.cu -o matrix_mul -I /usr/local/cuda/samples/common/inc

#Compile the new smc source code
smc: matrix_mul_smc.cu
	nvcc matrix_mul_smc.cu -o matrix_mul_smc -I /usr/local/cuda/samples/common/inc  -I ../smc/

#Source-to-source transform of matrix_mul.cu using smc
matrix_mul_smc.cu: matrix_mul.cu
	smc matrix_mul.cu -- --cuda-host-only -I /usr/local/cuda/samples/common/inc

clean:
	rm matrix_mul_smc.cu *.o matrix_mul matrix_mul_smc

# RBM Optimized for CUDA
__Author:__ Randall Pittman

This project develops an optimized RBM delta product implementation
for GPU. Multiple techniques are provided to demonstrate some of the
different experimental implementations for both GPU RBM and the delta
product. 

## Folder Description

data: contains the datasets for training the RBM.

src: The source code for the various implementations of RBM.

include: .h files for the source code.

obj: .o files constructed during compilation. 


## Different Implementations

There are several different implementations of the RBM algorithm along
with its delta product optimization, each of which is listed in the
form:

_ID : filename_

In this list, _ID_ is the text used when running the primary
executable to select a specific technique. _filename_ is the .cu or
.cpp file that contains the implementation for the
technique.


### cpu: cpu_baseline.cpp

A cpu implementation of RBM that can be used to roughly compare GPU vs
CPU performance.

### baseline : rbm_baseline.cu


A basic version of RBM implemented for the GPU. Uses dynamic
parallelism in CUDA kernels to accomplish parallel matrix
multiplication.

### dynamic_cublas : rbm_dynamic_cublas.cu


Similar to "baseline", but uses cublas matrix-vector multiplication
instead of a manual implementation. 

### delta_product : rbm_delta_product.cu


Similar to "dynamic_cublas", but includes the delta product
optimization. 

### matrix : rbm_matrix.cu


RBM implemented using cublas for matrix multiplication on
GPU. Generally much more efficient than its baseline.cu counterpart. 

### dp_matrix : rbm_dp_matrix.cu


RBM implemented using cublas for some matrix multiplication, and
optimized with the delta product matrix multiplication where
possible.

__NOTE:__ The best performing design for the baseline is
rbm_matrix.cu, and the best for delta product is rbp_dp_matrix.cu. 


## Compilation

Type "make" in the uppermost directory to build executable "rbm".

__NOTE:__ The makefile requires that the cublas libraries be present at path: 
/usr/local/cuda/lib64/
If this is not the case for your system, change line 12 of the
makefile to point to the correct path!


## Execution


The RBM can be run manually as follows:
./rbm train_N batchsize learning_rate k epoch dataset option n_visible n_hidden weightcost

Note that all parameters are required. The parameters are defined as
follows: 

train_N: positive integer
Number of images to train per epoch.

batchsize: positive integer
Number of images to train before updating the weight matrix.

learning_rate: positive float  1
Learning rate for RBM algorithm.

k: positive integer
Number of gibbs sampling steps to run. Higher values have a heavy
impact on performance.

epoch: positive integer
Number of epochs for RBM algorithm.

dataset: One of {mnist, flipmnist, cal101, newsgroup, micro-norb}
Which dataset to use when training the RBM.

option: One of {cpu, baseline, dynamic_cublas, delta_product, matrix, dp_matrix}
Determines which implementation of RBM to use when training. 

n_visible: positive integer
Number of visible nodes to use when training.

n_hidden: positive integer
Number of hidden nodes to use when training.

weightcost: positive float
Used when updating the weight matrix.

Example:
./rbm 10000 100 0.01 10 15 mnist dp_matrix 784 500 0.0001

Note that "cpu" is much slower than GPU implementations. When running
using cpu (and you don't have hours to spare), k and epochs should be
low. For example:
./rbm 10000 100 0.01 2 4 mnist cpu 784 500 0.0001


### Output

The final result of the computation is the weight matrix and the
visible and hidden bias arrays. These are not currently saved, since
the primary purpose of this code is to measure the performance of each
technique with different datasets.

If you wish to save the matrix, hbias and vbias arrays, go to include/constants.h and uncomment the SAVE_WEIGHTS line. Remember to run _make clean_ before rerunning _make_. The .dat files will be placed in the dat_files folder. Run ./move_dats.sh to rename these automatically to a numbered version so they won't be overwritten on the next run. 

After every epoch is completed, a short sample of these 3 arrays is
printed to stdout. This helps to verify that computation is proceeding
correctly. 

Every implementation prints a line like the following:
Training time: x

where x is the time required to train the RBM. Note that this does not
include the time required to load and preprocess data, since IO is not
what is being measured here.



### Performance testing

The script "generate_results.sh" automatically constructs the results
of the delta product optimization on the "matrix" implementation. That
is, it prints the speedup when running rbm_matrix.cu (matrix) as
a starting point, and comparing to rbm_dp_matrix.cu (dp_matrix).

To measure the performance of other techniques, simply run the code
with the specified technique (e.g. "cpu") and observe the resulting
training time that is print to stdout. 


### Notes on Bugs and Issues

rbm_dp_matrix.cu crashes when BIT_CODING is enabled in include/constants.h. Reason unknown. 

The resulting weight matrix and hbias and vbias arrays are slightly
different between certain CPU and GPU implementations. This difference
can be primarily attributed to the different random number generators
used between the CPU and GPU.


### Future Designs

The rbm_dp_matrix.cu code is the fastest delta product code. However, when
compiled for floating point code, it runs around the same speed as
rbm_matrix.cu, the baseline. Since double precision has speedups
around 2-3X, this is a rather disappointing result for floats.

My nvvp profiles of the primary dpmm_kernel indicate that it is memory
bound, not compute bound. This is despite the fact that using the
weight matrix and its transpose in the code has made all global
accesses nicely coalesced.

I still think there is a better layout for this kernel. Here's a
final concept kernel.

Instead of using aggregated-warp-filtering (AWF) to merge all the
diff indices to the beginning of the diff-array, divide the
diff-array into 32 sections. Each sub-array starts with the indices
that contain a difference only in that section. For example, if
indices 1, 11, 25, 54, 67, 82 are different and the total length is
1024, then each subarray has size 1024/32 = 32, so it is stored as:

```
index of diff-array: 0  1   2    3   4  ...  32  33 34 ... 64  65  66  67, ...
diff-array         : 1, 11, 25, -1, -1, ..., 54, -1 -1 ... 67, 82, -1, -1, ...
```

The first step for this modification is to modify the diff-array
creation code and the size of this array. AWF needs a global pointer
that defines the count of differences found. The easiest way to get
this pointer (I think) is to add an extra element at the beginning of
each subset. The first element is the counter that is sent to AWF, and
it automatically gets set to the number of diffs found in its
section. Yes, this will make the allocation of the array a bit
confusing, requiring a bit of arithmetic.

The next stage is modifying the dpmm_kernel....

Assume the batch layers are being multiplied by W stored in row-major
format. This is always possible, since the transpose is stored as well
and can be used to satisfy row-major. Assume W has _m_ rows and _n_
columns. 

Allocate grid of threads with x dimensions of 32 threads per block
with ceil(n/32) blocks, and y dimensions 32 threads per block with 1
block.

Each warp starts at row sublen * threadIdx.y of W, where sublen is the
length of each diffs-array subsection. This corresponds to index
(sublen + 1) * threadIdx.y of the diffs-array. Each warp reduces (via
addition) atomically its length 32 section of W into shared memory
(initialized to 0). The rows each warp should reduce into shared
memory is given by the fancy new diffs_array that contains the number
of diffs at (sublen + 1) * threadIdx.y, and it can then access that
number of diffs from the indices after this location, which provide
the rows of W that need to be reduced into shared memory (atomic
addition is needed I think, not sure if warps within the same block
conflict).

However, after this reduction only 1 element of the batch will have
been completed. So the kernel could be expanded over all these
batch elements in parallel (perhaps by changing y number of blocks to
batch size). The GPU probably won't be saturated well without this
expansion. 

This concept for doing the reduction maximizes weight matrix parallel
read performance, coalesces memory accesses, uses shared memory for
reduction variable, and parallelizes the task of iterating through the
diffs array, which is presently being done serially by each thread.

Note that it's possible I've overlooked something in this design that
would make the shared memory or some other part infeasible.

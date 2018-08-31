#!/bin/bash
#Author: Randall Pittman

# This file measures the speedup from the cublas-matrix baseline to
# the delta product version over multiple values of K
# NOTE: The script may appear to be stalling, but it is actually
# running RBM, which can take several minutes to complete!

#The parameters are ordered this way to make it easier to read from the RBM paper
# $1 = dataset
# $2 = train_N
# $3 = nvisible
# $4 = epochs
# $5 = K
generate() {
    echo "TEST CASE FOR: dataset = $1, train_N = $2, nvisible = $3, epochs = $4, K = $5";
    #baseTime=`./rbm $2 100 0.01 $5 $4 $1 matrix $3 500 0.0001 | grep "Training time:*" | grep -o "[0-9]*\.[0-9]*"`;
    #echo "matrix time: $baseTime";
    dpTime=`./rbm $2 100 0.01 $5 $4 $1 dp_matrix $3 500 0.0001 | grep "Training time:*" | grep -o "[0-9]*\.[0-9]*"`;
    echo "dp time: $dpTime";
    #echo Speedup: `python -c "print(round($baseTime/$dpTime, 3))"`;
}

for K in 15; do 
    generate mnist 50000 784 20 $K;
    generate flipmnist 50000 784 20 $K;
    generate cal101 4100 784 300 $K;
    generate newsgroup 8500 100 100 $K;
    generate micro-norb 15000 1024 100 $K;
done

exit 0;

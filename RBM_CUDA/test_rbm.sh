
#!/bin/bash
#Author: Randall Pittman

#The purpose of this script is to make it easier to run different
#techniques with different values of k and epochs.
#For example, to run dp_matrix with k=6 and epoch=20, just run:
# ./test_rbm.sh dp_matrix 6 20

#This script requires 1 parameter: the technique to use (dp_matrix, matrix, ...)
#The optional second and third parameters are the value of k and the number of epochs.
#Both k and number of epochs default to 5
#Uses the mnist dataset to train. 

#For reference:
#./rbm <train_N> <batchsize> <learning_rate> <k> <epoch> <filename> <option> <n_visible> <n_hidden> <weightcost>

K=10;
EPOCHS=5;

if [[ -z $1 ]]; then
    echo "Expected algorithm specification string, one of {baseline, cpu, matrix, delta_product, dp_matrix, dynamic_cublas}";
    exit 1;
fi

if [[ -z $2 ]]; then
    echo "k = $K";
else
    K=$2;
fi

if [[ -z $3 ]]; then
    echo "epoch = $EPOCHS";
else
    EPOCHS=$3;
fi

# NOTE: batch size must now be a power of 2 for the newer dpvm style kernel. 
# nvprof ./rbm 10000 128 0.01 $K $EPOCHS mnist $1 784 500 0.0001;
# /usr/local/cuda-8.0/bin/nvprof ./rbm 10000 128 0.01 $K $EPOCHS mnist $1 784 500 0.0001;

./rbm 10000 100 0.01 $K $EPOCHS mnist $1 784 500 0.0001;
echo "streams=$NUM_STREAMS, k=$K, epochs=$EPOCHS, technique=$1";
exit 0;


# test_rbm.sh
# ./rbm 10000 100 0.01 1 10 mnist matrix 784 500 0.0001
# ./rbm 50000 100 0.01 1 20 mnist matrix 784 500 0.0001

#!/bin/bash

for base in "hbias" "vbias" "weight_matrix"; do
    if [[ -z ${base}.dat ]]; then
	echo "${base}.dat not found";
    else
	count=1;
	full_name=${base}_${count}.dat;
	while [[ -f $full_name ]]; do
	    let "count=count+1";
	    full_name=${base}_${count}.dat;
	done
	mv ${base}.dat ${full_name};
    fi
done

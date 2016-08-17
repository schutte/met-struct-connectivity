#!/bin/sh

folds=$1
seed=$2

run_all_classes() {
	local method_template=$1
	local folds=$2
	local seed=$3

	for classes in 'hc,ftld'; do
		struct_seed=`perl -e 'print int(rand() * 100000)'`
		local method=`printf "$method_template" "$struct_seed"`

		class0=${classes%,*}
		class1=${classes#*,}

		echo -n "[ -f 'results/${class0}${class1}_${seed}_${folds}_${method}.csv' ] || "
		echo "python classify.py $method $class0 $class1 $folds $seed"
	done
}

run_all_classes unweighted_tfphammers_narrow $folds $seed
run_all_classes zero_tfphammers_narrow $folds $seed

for sc_type in '' '_normsc' '_fa' '_diff_normsc' '_diff_fa'; do
	for weighting in 'exp0.01' 'exp0.02' 'exp0.05' \
			'exp0.1' 'exp0.2' 'exp0.4' 'exp0.8' 'exp1' \
			'pow1' 'pow2' 'pow3'; do
		run_all_classes ${weighting}${sc_type}_tfphammers_narrow $folds $seed
		for i in `seq 1 10`; do
			run_all_classes ${weighting}${sc_type}_tfphammers_narrow_randstruct%d $folds $seed
		done
	done
done

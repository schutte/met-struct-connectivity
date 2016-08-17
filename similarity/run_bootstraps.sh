#! /bin/sh

set -e

echo real
python estimate.py zero_simhammers all

for i in `seq 1 1000`; do
	echo $i
	python estimate.py zero_simhammers_bootstrap$i all
done

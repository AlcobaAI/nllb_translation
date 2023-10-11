#!/bin/bash
#SBATCH --account=rrg-mageed
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:59:00
#SBATCH --mem=30gb

module load gcc/9.3.0 arrow/10.0.1 python/3.8 scipy-stack

source ~/ENV38_default/bin/activate

file=$1
output=$2
format=$3
column=$4
replace=$5
split_stc=$6
wzcuda=$7

python translate_nllb.py --file $file --output $output \
                        --data-format-in $format --data-format-out $format \
                        --translate-column $column --replace $replace \
                        --split-sentences $split_stc --with-cuda $wzcuda
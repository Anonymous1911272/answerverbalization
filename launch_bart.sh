#!/usr/bin/env/ bash

declare -a datasets=("VANiLLa" "VQuAnDa" "ParaQA" "VANiLLa VQuAnDa" "VANiLLa ParaQA" "VQuAnDa ParaQA" "VANiLLa VQuAnDa ParaQA")
declare -a tasks=("AV" "AV RQ")



for seed in "1234" "42" "158"
do

	for dataset in "${datasets[@]}"
	do

		for ratio in "100" "80" "60" "40" "20"
		do
			for task in "${tasks[@]}"
			do 

				CUDA_VISIBLE_DEVICES=1 python train.py --seed "$seed" --exp_name run"$seed" --training_datasets $dataset --training_ratio "$ratio" --lr "0.00001" --tasks $task --model "BartForConditionalGeneration" --tokenizer "BartTokenizer" --config "facebook/bart-base" --cache_dir "data/cmjt4501/" --lowercase --use_cuda

			done

		done
	done


done

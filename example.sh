#!/usr/bin/env/ bash


CUDA_VISIBLE_DEVICES=0 python train.py --seed "78782" --exp_name run78782 --training_datasets "ParaQA" --training_ratio "10" --tasks "AV" --model "BartForConditionalGeneration" --tokenizer "BartTokenizer" --config "facebook/bart-base" --cache_dir "data/cmjt4501/" --lowercase --use_cuda

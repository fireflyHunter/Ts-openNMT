#!/usr/bin/env bash
path_to_data=./data
path_to_model=./models
path_to_opennmt=../OpenNMT-py
python $path_to_opennmt/preprocess.py -dynamic_dict -share_vocab -train_src $path_to_data/good_train_source.txt -train_tgt $path_to_data/good_train_target.txt -valid_src $path_to_data/good_val_source.txt -valid_tgt $path_to_data/good_val_target.txt -save_data $path_to_data/processed/

CUDA_VISIBLE_DEVICES=2 python $path_to_opennmt/train.py -gpuid 0 -train_steps 20000 -data $path_to_data/processed/ -save_model $path_to_data/model -layers 1 -word_vec_size 300 -share_embeddings -rnn_size 300 -copy_attn -copy_attn_force

model=$(for i in $path_to_data/model*; do printf '%s\n' "$i"; break; done)
python $path_to_opennmt/translate.py -model $model -src $path_to_data/valid_self_original_no_cands_source.txt -gpu 0 -beam_size 30 -output $path_to_data/generated_output -n_best 1

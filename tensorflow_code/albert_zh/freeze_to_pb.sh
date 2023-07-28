#!/usr/bin/env bash
python freeze_graph.py \
    -bert_model_config "./pre_trained_model/albert_tiny_489k/albert_config_tiny.json" \
    -model_dir "./outputs/selfdata_albert_tiny_489k_output_epoch3" \
    -model_pb_dir "./pb_model_dir/albert_tiny_489k.pb" \
    -max_seq_len 128 \
    -num_labels 97

#!/usr/bin/env bash
python freeze_graph.py \
    -bert_model_dir "../pretrain_model/chinese_L-12_H-768_A-12" \
    -model_dir "./outputs/selfdata_bert_base_zh_output_epoch12" \
    -model_pb_dir "./pb_model_dir" \
    -max_seq_len 128 \
    -num_labels 97

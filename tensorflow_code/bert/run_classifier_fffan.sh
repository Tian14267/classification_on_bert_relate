#!/usr/bin/env bash
TASK_NAME="selfdata"
BERT_DIR="./pretrain_models/chinese_L-12_H-768_A-12"
CURRENT_DIR="./outputs"

echo "Start running..."

#:<<EOF
CUDA_VISIBLE_DEVICES=0 python run_classifier_fffan.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir="./data/cnews" \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_epochs=2.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_bert_base_zh_output_epoch2/
#EOF


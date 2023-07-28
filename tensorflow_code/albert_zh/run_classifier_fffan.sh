#!/usr/bin/env bash
TASK_NAME="selfdata"
ALBERT_TINY_DIR="./pre_trained_model/albert_tiny_489k"
CURRENT_DIR="./outputs"

echo "Start running..."

#:<<EOF
CUDA_VISIBLE_DEVICES=0 python run_classifier_fffan.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir="./data/self_data" \
  --vocab_file=$ALBERT_TINY_DIR/vocab.txt \
  --bert_config_file=$ALBERT_TINY_DIR/albert_config_tiny.json \
  --init_checkpoint=$ALBERT_TINY_DIR/albert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --num_train_epochs=15.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_albert_tiny_489k_output_epoch15/
#EOF

:<<EOF
python run_classifier_fffan_google.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir="./data/self_data" \
  --vocab_file=$ALBERT_TINY_DIR/vocab.txt \
  --bert_config_file=$ALBERT_TINY_DIR/albert_config_tiny_g.json \
  --init_checkpoint=$ALBERT_TINY_DIR/albert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --num_train_epochs=3.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_albert_tiny_zh_google_output_epoch3/
EOF
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 \
      run_roberta_class_fffan_distributed.py \
          --model_path "/data/fffan/0_experiment/7_Bert/00_models/chinese_roberta_wwm_large_ext_pytorch" \
          --train_path "/data/fffan/0_experiment/7_Bert/02_classify/data/messanswer_data_train/messanswer_train.txt" \
          --dev_path "/data/fffan/0_experiment/7_Bert/02_classify/data/messanswer_data_train/messanswer_val.txt" \
          --test_path "/data/fffan/0_experiment/7_Bert/02_classify/data/messanswer_data_train/messanswer_test.txt" \
          --class_list_path "/data/fffan/0_experiment/7_Bert/02_classify/data/messanswer_data_train/messanswer_label.txt" \
          --output_dir "./saved_models/roberta_large_distributed/" \
          --train_batch_size 80 \
          --eval_batch_size 8 \
          --num_train_epochs 20 \
          --sequence_len 512
export PYTHONUNBUFFERED=1

########
##  Bert
#nohup python run_bert_class_fffan.py > nohup_bert_base_s256 2>&1 &

#######
##  AlBert
#nohup python run_albert_class_fffan.py > nohup_albert_base_s256 2>&1 &

######
##  RoBerta
nohup python run_roberta_class_fffan.py > nohup_roberta_wwm_s256_THUCNews 2>&1 &
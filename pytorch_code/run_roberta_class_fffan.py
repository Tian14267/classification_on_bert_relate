# coding: UTF-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
import torch.nn as nn
import numpy as np
from train_eval import train
import argparse
from models.roberta_models import RobertaModel, BertTokenizer
from utils import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.model_path)
        for param in self.roberta.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outs = self.roberta(context, attention_mask=mask)
        out = self.fc(outs["pooler_output"])
        return out


def main():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model_name', type=str, default="roberta_base", help='')
    parser.add_argument('--model_path', type=str, default="./pretrain_model/roberta_chinses_wwm_ext", help='')
    parser.add_argument('--train_path', type=str, default="./data/THUCNews/train.txt", help='')
    parser.add_argument('--dev_path', type=str, default="./data/THUCNews/dev.txt", help='')
    parser.add_argument('--test_path', type=str, default="./data/THUCNews/test.txt", help='')
    parser.add_argument('--class_list_path', type=str, default="./data/THUCNews/class.txt", help='')
    parser.add_argument('--class_list', type=str, default=[], help='')
    parser.add_argument('--save_path', type=str, default="./saved_models/sql_256/THUCNews/", help='')
    parser.add_argument('--device', type=str, default="cuda", help='')
    parser.add_argument('--label_map', type=bool, default=False, help='是否进行label映射ID')

    parser.add_argument('--require_improvement', type=int, default=1000, help='')
    parser.add_argument('--num_epochs', type=int, default=20, help='')
    parser.add_argument('--num_classes', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=80, help='')
    parser.add_argument('--sequence_len', type=int, default=256, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='')
    parser.add_argument('--hidden_size', type=int, default=768, help='')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if os.path.exists(args.class_list_path):
        args.class_list = [x.strip() for x in open(args.class_list_path).readlines()]  # 类别名单
    else:
        train_class_list = get_class_list(self.train_path)
        dev_class_list = get_class_list(self.dev_path)
        test_class_list = get_class_list(self.test_path)

        assert len(list(set(train_class_list) - set(dev_class_list))) == 0
        assert len(list(set(train_class_list) - set(test_class_list))) == 0
        args.class_list = train_class_list
        write_file(args.class_list, args.class_list_path)

    if len(args.class_list) != args.num_classes:
        args.num_classes = len(args.class_list)
        print("###  类别数：",args.num_classes)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    #if args.device == "cuda":
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_list = [x.strip() for x in open(args.class_list_path).readlines()]
    args.num_classes = len(classes_list)
    #tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(args,tokenizer)
    train_iter = build_iterator(train_data, args)
    dev_iter = build_iterator(dev_data, args)
    test_iter = build_iterator(test_data, args)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = Model(args).to(args.device)
    train(args, model, train_iter, dev_iter, test_iter)


if __name__ == '__main__':
    main()

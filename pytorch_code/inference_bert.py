# coding: UTF-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from models.bert_models import BertModel, BertTokenizer
from utils import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)  ### 加载相关信息，如vocab和config等
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


class ModelInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Chinese Text Classification')
        parser.add_argument('--model_name', type=str, default="bert", help='')
        parser.add_argument('--bert_path', type=str, default="./saved_models/bert_out/bert_128", help='')
        parser.add_argument('--class_list_path', type=str, default="./data/self_data/class.txt", help='')
        parser.add_argument('--class_list', type=str, default=[], help='')
        parser.add_argument('--device', type=str, default="cpu", help='')
        parser.add_argument('--label_map', type=bool, default=True, help='是否进行label映射ID')
        parser.add_argument('--num_classes', type=int, default=0, help='')
        parser.add_argument('--sequence_len', type=int, default=128, help='')
        parser.add_argument('--hidden_size', type=int, default=768, help='')

        self.args = parser.parse_args()

        self.args.class_list = [x.strip() for x in open(self.args.class_list_path).readlines()]  # 类别名单
        self.args.num_classes = len(self.args.class_list)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        # if args.device == "cuda":
        if self.args.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)

        # train
        self.model = Model(self.args).to(self.device)
        ####  torch  模型加载
        params = torch.load(os.path.join(self.args.bert_path, "pytorch_model.bin"))  # 加载参数
        self.model.load_state_dict(params)  # 应用到网络结构中


    def build_input_data(self,content):
        token = self.tokenizer.tokenize(content)
        token = [CLS] + token
        #mask = []
        token_ids = self.tokenizer.convert_tokens_to_ids(token)

        if self.args.sequence_len:
            if len(token) < self.args.sequence_len:
                mask = [1] * len(token_ids) + [0] * (self.args.sequence_len - len(token))
                token_ids += ([0] * (self.args.sequence_len - len(token)))
            else:
                mask = [1] * self.args.sequence_len
                token_ids = token_ids[:self.args.sequence_len]

        token_ids_tensor = torch.LongTensor([token_ids]).to(self.device)
        mask_ids_tensor = torch.LongTensor([mask]).to(self.device)

        return [token_ids_tensor, mask_ids_tensor]


    def predict(self, texts):
        ####  单条推理
        token_inputs = self.build_input_data(texts)
        with torch.no_grad():
            outputs = self.model(token_inputs)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        predic_class = self.args.class_list[predic[0]]
        return predic_class


    def evaluate(self,path):
        ######   通过推理方式，测试准确率。验证推理方案
        all_num = 0
        all_right_num = 0
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            all_num = len(lines)
            for line in tqdm(lines):
                lin = line.strip()
                content, label = lin.split('\t')
                token_inputs = self.build_input_data(content)
                with torch.no_grad():
                    outputs = self.model(token_inputs)
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
                    if self.args.label_map:
                        predic = self.args.class_list[predic]
                    else:
                        predic = str(int(predic))

                    if predic == label:
                        all_right_num = all_right_num + 1

        assert all_num!=0

        predict_rate = all_right_num / all_num

        print(predict_rate)


    def predict_mutil(self, path):
        ####  批量测试推理效率和耗时
        start_time = time.time()
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                content, label = lin.split('\t')
                token_inputs = self.build_input_data(content)
                with torch.no_grad():
                    outputs = self.model(token_inputs)
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                predic_class = self.args.class_list[predic[0]]
        end_time = time.time()
        print("###  耗时：", end_time - start_time)


if __name__ == '__main__':
    MI = ModelInit()
    ####  单挑推理    ###################################################
    #texts = "多家网站关闭车票转让信息"
    #out = MI.predict(texts)
    #print(out)
    ####  批量测试推理效率和耗时    #######################################
    #out = MI.predict_mutil("./data/self_data/test_1k.txt")
    ####  测试准确率，验证推理方案    #####################################
    MI.evaluate("./data/THUCNews/test.txt")

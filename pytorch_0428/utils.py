# coding: UTF-8
import torch
from tqdm import tqdm
import time
import random
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config,tokenizer):
    class_dict = {}
    for i, label in enumerate(config.class_list):
        class_dict[label] = i
    def load_dataset(path, seq_len=32):
        contents = []
        error_list = []
        with open(path, 'r', encoding='UTF-8') as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                try:
                    lin = line.strip()
                    if not lin:
                        continue
                    #content, label = lin.split('\t')
                    label,content = lin.split('\t')
                    if config.label_map:
                        label = class_dict[label]
                    token = tokenizer.tokenize(content)
                    token = [CLS] + token
                    mask = []
                    token_ids = tokenizer.convert_tokens_to_ids(token)

                    if seq_len:
                        if len(token) < seq_len:
                            mask = [1] * len(token_ids) + [0] * (seq_len - len(token))
                            token_ids += ([0] * (seq_len - len(token)))
                        else:
                            mask = [1] * seq_len
                            token_ids = token_ids[:seq_len]
                    contents.append((token_ids, int(label), seq_len, mask))
                except:
                    error_list.append(line)
        if error_list:
            print("#####  问题数据样例：", error_list[0])
            print("#####  问题数据量：",len(error_list))
        return contents
    #train = load_dataset(config.train_path, config.sequence_len)
    #dev = load_dataset(config.dev_path, config.sequence_len)
    test = load_dataset(config.test_path, config.sequence_len)
    train = test
    dev = test
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        random.shuffle(self.batches)  ### 随机打乱数据顺序
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_class_list(file_path):
    class_list = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            class_list.append(line.replace("\n", "").split("	")[1])
    class_list = list(set(class_list))
    return class_list

def read_files(file):
    with open(file,"r",encoding="utf-8") as f:
    #with open(file, "r", encoding="GBK") as f:
        lines = f.readlines()
        lines_out = []
        for line in lines:
            line = line.replace("\n","")
            lines_out.append(line)
    f.close()
    return lines_out

def write_file(lines,file):
    with open(file,"w",encoding="utf-8") as f:
        for line in lines:
            line = line + "\n"
            f.write(line)
    f.close()
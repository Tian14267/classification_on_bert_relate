# coding=utf-8
'''
@Software:PyCharm
@Time:2025/04/29
@Author: fffan
'''

from __future__ import absolute_import, division, print_function

import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import logging
import shutil
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from collections import namedtuple
from torch.utils.data import (DataLoader, RandomSampler, Dataset)
from utils import *
from torch.nn import CrossEntropyLoss
import collections
from sklearn import metrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import RobertaModel, BertTokenizer,RobertaTokenizer




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

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


class PreTrainingDataset(Dataset):
    """ 实际的供给 Dataloader 的数据类 """

    def __init__(self):
        self.data = []

    def add_instance(self, features: collections.OrderedDict):
        self.data.append((
            features["input_ids"],
            features["segment_ids"],
            features["input_mask"],
            features["masked_lm_ids"],
            features["next_sentence_labels"]
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label = self.data[index]
        return input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label


class PregeneratedDataset(object):
    def __init__(self, file_list, tokenizer):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        logger.info('总数据量: {}'.format(len(file_list)))
        self.input_ids = []
        self.segment_ids = []
        self.input_masks = []
        self.label_ids = []
        print("#####  开始读取数据...",)
        random.shuffle(file_list)  ### 随机打乱数据顺序

        for i, one_data in enumerate(tqdm(file_list)):
            data_1,label = self._to_tensor([one_data])
            self.input_ids.append(data_1[0])
            self.segment_ids.append(data_1[1])
            self.input_masks.append(data_1[2])
            self.label_ids.append(label)

        self.data_size = len(self.input_ids)

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas])
        y = torch.LongTensor([_[1] for _ in datas])

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2]*[1] for _ in datas])
        mask = torch.LongTensor([_[3] for _ in datas])

        x = torch.squeeze(x, dim=0)
        seq_len = torch.squeeze(seq_len, dim=0)
        mask = torch.squeeze(mask, dim=0)

        y = torch.squeeze(y, dim=0)
        return (x, seq_len, mask), y


    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        return (
            self.input_ids[item].clone().detach(),
            self.input_masks[item].clone().detach(),
            self.segment_ids[item].clone().detach(),
            self.label_ids[item].clone().detach()
        )

def get_time():
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return current_time

def get_parser():
    parser = argparse.ArgumentParser()
    #########################
    parser.add_argument("--train_path",
                        default="/data/fffan/01_experiment/08_Bert/02_classify/data/messanswer_data_train/messanswer_test.txt",
                        type=str)
    parser.add_argument("--dev_path",
                        default="/data/fffan/01_experiment/08_Bert/02_classify/data/messanswer_data_train/messanswer_val.txt",
                        type=str)
    parser.add_argument("--test_path",
                        default="/data/fffan/01_experiment/08_Bert/02_classify/data/messanswer_data_train/messanswer_test.txt",
                        type=str)

    parser.add_argument('--class_list_path', type=str,
                        default="/data/fffan/01_experiment/08_Bert/02_classify/data/messanswer_data_train/messanswer_label.txt",
                        help='')
    # Required parameters
    parser.add_argument("--model_path", default="/data/fffan/01_experiment/08_Bert/00_models/chinese_roberta_wwm_large_ext_pytorch",
                        type=str)
    parser.add_argument("--output_dir", default="./output_dir/pretrain_fffan", type=str)
    parser.add_argument('--class_list', type=str, default=[], help='')
    parser.add_argument('--num_classes', type=int, default=0, help='')
    parser.add_argument('--label_map', type=bool, default=True, help='是否进行label映射ID')
    parser.add_argument('--hidden_size', type=int, default=1024, help='')

    # Other parameters
    parser.add_argument("--save_model_number",
                        default=5,
                        type=int, help="The maximum total input sequence length ")
    parser.add_argument("--sequence_len",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece \n"
                             " tokenization. Sequences longer than this will be truncated, \n"
                             "and sequences shorter than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        # default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-1,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus. torch==1.13.1 for local_rank, and torch>1.13.1 for local-rank")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing \n"
                             "a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')

    # Additional arguments
    parser.add_argument('--eval_step', type=int, default=5)


    args = parser.parse_args()

    return args

def save_model(global_step, model, output_dir):
    logging.info("** ** * Saving  model ** ** * ")
    prefix = f"step_{global_step}"
    logging.info("** ** * Saving  model ** ** * ")
    model_name = "{}_{}".format(prefix, WEIGHTS_NAME)  ###  step_5_pytorch_model.bin
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, model_name)
    torch.save(model_to_save.state_dict(), output_model_file)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    #model_to_save.config.to_json_file(output_config_file)
    return output_model_file


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def evaluate(dev_loader,model,local_rank, args):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    print("######   开始进行验证！")
    with torch.no_grad():
        for dev_step, batch in enumerate(tqdm(dev_loader, desc="# Dev Iteration", ascii=True)):
            #####
            batch = tuple(t.cuda(local_rank, non_blocking=True) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model((input_ids, segment_ids, input_mask))

            loss = F.cross_entropy(outputs, label_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            #loss = loss.detach().cpu().numpy()

            #loss = reduce_mean(loss, args.nprocs).detach().cpu().numpy()
            #loss_total += loss
            loss_total += loss.item()  # 直接使用 .item() 避免转为numpy
            labels = label_ids.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            #"""

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(dev_loader)


def train(train_loader, dev_loader, model,global_step,all_save_model_list,
          optimizer, local_rank, args,
          epoch,best_model_info,scheduler):
    # switch to train mode
    model.train()
    #all_save_model_list = []
    for step, batch in enumerate(tqdm(train_loader, desc="# Train", ascii=True)):
        #####
        batch = tuple(t.cuda(local_rank, non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        #print("label_ids: ",label_ids.shape)
        if input_ids.size()[0] != args.train_batch_size:
            continue

        outputs = model((input_ids, segment_ids, input_mask))

        loss = F.cross_entropy(outputs, label_ids)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        ####  多卡loss均衡
        #loss = reduce_value(loss, average=True)
        loss = reduce_mean(loss, args.nprocs).detach().cpu().numpy()

        optimizer.step()
        optimizer.zero_grad()  ###  把该优化器所管理的参数的梯度清零。
        model.zero_grad()  ###  把模型里所有可训练参数的梯度清零。
        global_step += 1

        lr_now = optimizer.param_groups[0]["lr"]

        #"""
        if (global_step + 1) % args.eval_step == 0 and local_rank == 0:
            train_labels = label_ids.data.cpu().numpy()
            train_predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            train_acc = metrics.accuracy_score(train_labels, train_predic)

            ######   eval
            dev_acc, dev_loss = evaluate(dev_loader, model, local_rank, args)
            # 更新学习率
            scheduler.step(dev_loss)
            if dev_acc > best_model_info["dev_best_acc"]:
                best_model_info["dev_best_acc"] = dev_acc
                best_model_info["best_model_step"] = global_step
                #save_path_in_epoch = os.path.join(config.save_path,
                #                                  config.model_name + "_pytorch_model_" + str(epoch) + ".bin")
                # print(save_path_in_epoch)
                #torch.save(model.state_dict(), save_path_in_epoch)
                improve = '*'
                last_improve = global_step
            else:
                improve = ''
            time_now = get_time()
            msg = ('{0} Epoch:{1} Iter: {2:>6},  Train Loss: {3:>5.2},  Train Acc: {4:>6.2%}, LR:{5}  Val Loss: {6:>5.2}, '
                   'Val Acc: {7:>6.2%}   {8}  \n').format(str(time_now),epoch,global_step, loss.item(),
                           train_acc,lr_now,dev_loss, dev_acc,  improve
                   )
            logger.info(msg)

            output_eval_file = os.path.join(args.output_dir, "log.txt")
            with open(output_eval_file, "a") as writer:
                writer.write(msg)
                logger.info("***** Eval results *****\n")
                writer.write("***** Eval results *****  \n")
                writer.write("Epoch = %s\n" % (str(epoch)))
                writer.write("global_step = %s\n" % (str(global_step)))
                for name,value in zip(("Eval_Acc","Eval_Loss"),(dev_acc,dev_loss)):
                    logger.info("  %s = %s", name, str(value))
                    writer.write("%s = %s\n" % (name, str(value)))

            # 保存最优模型
            ########################################################################
            if improve:
                output_model_file = save_model(global_step, model, args.output_dir)
                #####  保存的模型数量超过指定数量，删除模型  #############
                if len(all_save_model_list) == args.save_model_number:
                    os.system("rm -rf " + all_save_model_list[0])
                    print("####  删除模型：", all_save_model_list[0])
                    del all_save_model_list[0]
                ######################################################

                all_save_model_list.append(output_model_file)
        #"""
    if local_rank == 0:
        ######  复制模型
        if all_save_model_list:
            model_path_info = "/".join(all_save_model_list[-1].split("/")[:-1])+"/"+WEIGHTS_NAME
            print(all_save_model_list[-1])
            print(model_path_info)
            shutil.copyfile(all_save_model_list[-1], model_path_info)
    return global_step

def main_worker(local_rank, nprocs, args):
    dist.init_process_group(backend='nccl')


    ###########   初始化输入参数
    if os.path.exists(os.path.join(args.model_path, "vocab.txt")):
        os.system("cp " + os.path.join(args.model_path, "vocab.txt") + " " + args.output_dir)

    args.train_batch_size = int(args.train_batch_size / nprocs)

    if os.path.exists(args.class_list_path):
        args.class_list = [x.strip() for x in open(args.class_list_path).readlines()]  # 类别名单
    else:
        train_class_list = get_class_list(args.train_path)
        dev_class_list = get_class_list(args.dev_path)
        test_class_list = get_class_list(args.test_path)

        assert len(list(set(train_class_list) - set(dev_class_list))) == 0
        assert len(list(set(train_class_list) - set(test_class_list))) == 0
        args.class_list = train_class_list
        write_file(args.class_list, args.class_list_path)

    if len(args.class_list) != args.num_classes:
        args.num_classes = len(args.class_list)
        print("###  类别数：",args.num_classes)
    ############################################################

    ########    初始化模型
    model = Model(args)
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    #tokenizer = RobertaTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    print("#######  local_rank: ", local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    #################    预处理数据
    train_data, dev_data, test_data = build_dataset(args, tokenizer)
    train_dataset = PregeneratedDataset(train_data, tokenizer)
    dev_dataset = PregeneratedDataset(dev_data, tokenizer)
    test_dataset = PregeneratedDataset(test_data, tokenizer)

    total_train_examples = len(train_dataset)
    print("#####   训练数据量：", total_train_examples)
    ####  训练数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch_size,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)
    #####  验证数据
    dev_sampler = torch.utils.data.distributed.DistributedSampler(
        dev_dataset)

    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                               batch_size=args.eval_batch_size,
                                               num_workers=1,
                                               pin_memory=True,
                                               sampler=dev_sampler)

    #####  测试数据
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.eval_batch_size,
                                             num_workers=1,
                                             pin_memory=True,
                                             sampler=test_sampler)
    #########    计算模型参数信息
    #size = 0
    #for n, p in model.named_parameters():
    #    logger.info('n: {}'.format(n))
    #    logger.info('p: {}'.format(p.nelement()))
    #    size += p.nelement()
    #logger.info('Total parameters: {}'.format(size))
    ####################################################

    num_train_optimization_steps = int(total_train_examples / args.train_batch_size / args.num_train_epochs)
    #print("#####   训练数据总步数：", num_train_optimization_steps)
    if args.local_rank != -1:  ###  意味着当前代码不是在分布式训练环境下运行。
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size(
        ) * args.num_train_epochs

    #  optimizer  优化器设置
    #param_optimizer = list(model.named_parameters())
    #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #optimizer = BertAdam(optimizer_grouped_parameters,
    #                     lr=args.learning_rate,
    #                     warmup=args.warmup_proportion,
    #                     t_total=num_train_optimization_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 定义 ReduceLROnPlateau 调度器，当验证集损失在 5 个 epoch 内没有改善时，学习率乘以 0.1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    ##############################################
    cudnn.benchmark = True

    global_step = 0
    all_save_model_list = []
    best_model_info = {"dev_best_acc":0.0,"best_model_step":""}
    for one_epoch in range(int(args.num_train_epochs)):
        train_sampler.set_epoch(one_epoch)

        # train for one epoch
        global_step = train(train_loader, dev_loader, model, global_step,all_save_model_list, optimizer, local_rank,
              args, one_epoch,best_model_info,scheduler)

    ######   测试
    logger.info("####   开始测试。。。")
    test_acc, test_loss = evaluate(test_loader, model, local_rank, args)
    output_eval_file = os.path.join(args.output_dir, "log.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Test results *****")
        writer.write("***** Test results *****  \n")
        for name, value in zip(("Test_Acc", "Test_Loss"), (test_acc, test_loss)):
            logger.info("  %s = %s", name, str(value))
            writer.write("%s = %s\n" % (name, str(value)))
    ############################################################

def main():
    args = get_parser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    args.nprocs = torch.cuda.device_count()
    print("###########    ", args.nprocs)
    main_worker(args.local_rank, args.nprocs, args)


if __name__ == "__main__":
    main()
